import pandas as pd
import logging
from typing import Optional, Tuple
from binance.client import Client
from binance.exceptions import BinanceAPIException
import traceback
import time
import time
import datetime


logger = logging.getLogger("binance_utils")

def fetch_binance_klines(client, symbol: str, interval: str, limit: int = 1000, days: int = None) -> pd.DataFrame:
    """
    Загружает исторические данные с Binance.
    """
    try:
        all_klines = []
        
        # Сначала проверим, что можем получить данные
        test_klines = client.get_klines(symbol=symbol, interval=interval, limit=10)
        logger.info(f"Тест API: получено {len(test_klines)} свечей")
        
        if test_klines:
            first_time = datetime.datetime.fromtimestamp(test_klines[0][0] / 1000)
            last_time = datetime.datetime.fromtimestamp(test_klines[-1][0] / 1000)
            logger.info(f"Тестовые данные: от {first_time} до {last_time}")
        
        if days:
            intervals_per_day = {
                '1m': 1440,
                '3m': 480,
                '5m': 288,
                '15m': 96,
                '30m': 48,
                '1h': 24,
                '2h': 12,
                '4h': 6,
                '1d': 1
            }
            
            if interval in intervals_per_day:
                total_candles_needed = days * intervals_per_day[interval]
                logger.info(f"Запрос {total_candles_needed} свечей ({days} дней) для {symbol} с интервалом {interval}")
            else:
                total_candles_needed = limit
        else:
            total_candles_needed = limit
        
        # Получаем текущее время
        current_time = int(datetime.datetime.now().timestamp() * 1000)
        
        # Первый запрос - получаем самые свежие данные
        batch_size = min(1000, total_candles_needed)
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=batch_size
        )
        
        if not klines:
            logger.error(f"Не получены данные для {symbol}")
            return pd.DataFrame()
        
        all_klines = klines
        logger.info(f"Первый батч: получено {len(klines)} свечей")
        
        # Если нужно больше данных, делаем дополнительные запросы
        while len(all_klines) < total_candles_needed:
            # Берем timestamp первой (самой старой) свечи
            oldest_timestamp = all_klines[0][0]
            
            # Запрашиваем данные до этого времени
            batch_size = min(1000, total_candles_needed - len(all_klines))
            
            logger.debug(f"Запрос исторических данных до {datetime.datetime.fromtimestamp(oldest_timestamp/1000)}")
            
            older_klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=batch_size,
                endTime=oldest_timestamp - 1
            )
            
            if not older_klines:
                logger.info(f"Больше исторических данных не доступно для {symbol}")
                break
            
            logger.info(f"Получено еще {len(older_klines)} исторических свечей")
            
            # Добавляем старые данные в начало
            all_klines = older_klines + all_klines
            
            # Если получили меньше, чем запросили - достигли конца истории
            if len(older_klines) < batch_size:
                logger.info(f"Достигнут конец доступной истории для {symbol}")
                break
            
            # Задержка между запросами
            time.sleep(0.2)
        
        logger.info(f"Всего получено {len(all_klines)} свечей для {symbol}")
        
        # Преобразование в DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Конвертация типов
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Числовые колонки
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Удаляем дубликаты и сортируем
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        
        if len(df) > 1:
            time_range = df.index[-1] - df.index[0]
            logger.info(f"DataFrame для {symbol} создан: {len(df)} строк, период: {time_range.days:.1f} дней")
            logger.info(f"Данные от {df.index[0]} до {df.index[-1]}")
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка получения данных для {symbol}: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def get_usdt_balance(client: Client) -> float:
    """Получение баланса USDT"""
    try:
        account = client.get_account()
        for balance in account['balances']:
            if balance['asset'] == 'USDT':
                return float(balance['free'])
        return 0.0
    except Exception as e:
        logger.error(f"Ошибка получения баланса: {e}")
        return 0.0

def get_symbol_price(client: Client, symbol: str) -> Optional[float]:
    """Получение текущей цены символа"""
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        logger.error(f"Ошибка получения цены для {symbol}: {e}")
        return None

def get_lot_size(client: Client, symbol: str) -> Tuple[float, int]:
    """Получение минимального размера лота и точности для символа"""
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            return 0.001, 3
        
        for filter in info['filters']:
            if filter['filterType'] == 'LOT_SIZE':
                min_qty = float(filter['minQty'])
                step_size = float(filter['stepSize'])
                
                # Определение точности
                precision = 0
                temp = step_size
                while temp < 1:
                    temp *= 10
                    precision += 1
                
                return min_qty, precision
        
        return 0.001, 3
        
    except Exception as e:
        logger.error(f"Ошибка получения размера лота для {symbol}: {e}")
        return 0.001, 3

def place_market_order(client: Client, symbol: str, side: str, quantity: float) -> Optional[dict]:
    """Размещение рыночного ордера"""
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        logger.info(f"Размещен рыночный ордер: {symbol} {side} {quantity}")
        return order
    except BinanceAPIException as e:
        logger.error(f"Ошибка размещения ордера: {e}")
        return None

def place_stop_loss_order(client: Client, symbol: str, side: str, quantity: float, stop_price: float) -> Optional[dict]:
    """Размещение стоп-лосс ордера"""
    try:
        # Округление цены
        price_precision = get_price_precision(client, symbol)
        stop_price = round(stop_price, price_precision)
        
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='STOP_LOSS_LIMIT',
            timeInForce='GTC',
            quantity=quantity,
            stopPrice=stop_price,
            price=stop_price
        )
        logger.info(f"Размещен стоп-лосс: {symbol} {side} по цене {stop_price}")
        return order
    except BinanceAPIException as e:
        logger.error(f"Ошибка размещения стоп-лосса: {e}")
        return None

def place_take_profit_order(client: Client, symbol: str, side: str, quantity: float, price: float) -> Optional[dict]:
    """Размещение тейк-профит ордера"""
    try:
        # Округление цены
        price_precision = get_price_precision(client, symbol)
        price = round(price, price_precision)
        
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='LIMIT',
            timeInForce='GTC',
            quantity=quantity,
            price=price
        )
        logger.info(f"Размещен тейк-профит: {symbol} {side} по цене {price}")
        return order
    except BinanceAPIException as e:
        logger.error(f"Ошибка размещения тейк-профита: {e}")
        return None

def cancel_order(client: Client, symbol: str, order_id: str) -> bool:
    """Отмена ордера с проверкой существования"""
    try:
        # Сначала проверяем, существует ли ордер
        try:
            order_status = client.get_order(symbol=symbol, orderId=order_id)
            if order_status['status'] in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                logger.info(f"Ордер {order_id} для {symbol} уже в статусе {order_status['status']}, отмена не требуется")
                return True
        except BinanceAPIException as e:
            if e.code == -2013:  # Order does not exist
                logger.info(f"Ордер {order_id} для {symbol} не существует, отмена не требуется")
                return True
            # Другие ошибки проверки пропускаем и пытаемся отменить ордер
        
        # Отменяем ордер
        client.cancel_order(symbol=symbol, orderId=order_id)
        logger.info(f"Отменен ордер {order_id} для {symbol}")
        return True
        
    except BinanceAPIException as e:
        # Если ордер не найден - это не ошибка, он уже отменен или исполнен
        if e.code == -2011:  # Unknown order
            logger.info(f"Ордер {order_id} для {symbol} уже отменен или исполнен")
            return True
        else:
            logger.error(f"Ошибка отмены ордера: {e}")
            return False
    except Exception as e:
        logger.error(f"Ошибка отмены ордера: {e}")
        return False

def get_price_precision(client: Client, symbol: str) -> int:
    """Получение точности цены для символа"""
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            return 8
        
        for filter in info['filters']:
            if filter['filterType'] == 'PRICE_FILTER':
                tick_size = float(filter['tickSize'])
                precision = 0
                temp = tick_size
                while temp < 1:
                    temp *= 10
                    precision += 1
                return precision
        
        return 8
        
    except Exception as e:
        logger.error(f"Ошибка получения точности цены для {symbol}: {e}")
        return 8

# binance_utils.py

def get_order_book_data(client: Client, symbol: str, limit: int = 20) -> dict:
    """Получение данных стакана ордеров"""
    try:
        order_book = client.get_order_book(symbol=symbol, limit=limit)
        
        # Расчет метрик стакана
        bids = [{'price': float(bid[0]), 'quantity': float(bid[1])} for bid in order_book['bids']]
        asks = [{'price': float(ask[0]), 'quantity': float(ask[1])} for ask in order_book['asks']]
        
        bid_volume = sum(bid['quantity'] for bid in bids)
        ask_volume = sum(ask['quantity'] for ask in asks)
        
        # Расчет дисбаланса
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Расчет спреда
        best_bid = bids[0]['price'] if bids else 0
        best_ask = asks[0]['price'] if asks else 0
        spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
        spread_pct = spread * 100
        
        # Расчет глубины рынка
        bid_depth = sum(bid['price'] * bid['quantity'] for bid in bids)
        ask_depth = sum(ask['price'] * ask['quantity'] for ask in asks)
        
        # Расчет давления покупки/продажи
        buy_pressure = bid_volume / ask_volume if ask_volume > 0 else 0
        sell_pressure = ask_volume / bid_volume if bid_volume > 0 else 0
        
        return {
            'bids': bids,
            'asks': asks,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            'spread': spread,
            'spread_pct': spread_pct,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'best_bid': best_bid,
            'best_ask': best_ask
        }
    except Exception as e:
        logger.error(f"Ошибка получения данных стакана для {symbol}: {e}")
        return {}

def get_recent_trades(client: Client, symbol: str, limit: int = 100) -> pd.DataFrame:
    """Получение последних сделок"""
    try:
        trades = client.get_recent_trades(symbol=symbol, limit=limit)
        
        if not trades:
            return pd.DataFrame()
            
        df = pd.DataFrame(trades)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['price'] = df['price'].astype(float)
        df['qty'] = df['qty'].astype(float)
        df['quoteQty'] = df['quoteQty'].astype(float)
        
        # Определение агрессивной стороны (isBuyerMaker = False означает агрессивную покупку)
        df['aggressive_buy'] = ~df['isBuyerMaker']
        
        # Расчет объема агрессивных покупок/продаж
        df['buy_volume'] = df.apply(lambda x: x['qty'] if x['aggressive_buy'] else 0, axis=1)
        df['sell_volume'] = df.apply(lambda x: x['qty'] if not x['aggressive_buy'] else 0, axis=1)
        
        return df
    except Exception as e:
        logger.error(f"Ошибка получения последних сделок для {symbol}: {e}")
        return pd.DataFrame()

def calculate_vwap(client: Client, symbol: str, interval: str = "15m", limit: int = 100) -> float:
    """Расчет VWAP (Volume-Weighted Average Price)"""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        if not klines:
            return 0.0
            
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Расчет типичной цены
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Расчет VWAP
        df['vwap_value'] = df['typical_price'] * df['volume']
        vwap = df['vwap_value'].sum() / df['volume'].sum() if df['volume'].sum() > 0 else 0
        
        return vwap
    except Exception as e:
        logger.error(f"Ошибка расчета VWAP для {symbol}: {e}")
        return 0.0
    
def place_oco_order(client: Client, symbol: str, side: str, quantity: float, 
                   price: float, stop_price: float, stop_limit_price: float = None) -> Optional[dict]:
    """Размещение OCO ордера (One-Cancels-the-Other)"""
    try:
        # Округление цен согласно требованиям биржи
        price_precision = get_price_precision(client, symbol)
        price = round(price, price_precision)
        stop_price = round(stop_price, price_precision)
        
        if stop_limit_price is None:
            stop_limit_price = stop_price
        else:
            stop_limit_price = round(stop_limit_price, price_precision)
        
        # Базовые параметры для OCO ордера
        params = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'stopPrice': stop_price,
            'stopLimitPrice': stop_limit_price,
            'stopLimitTimeInForce': 'GTC'
        }
        
        # Используем непосредственно метод API
        endpoint = 'order/oco'
        order = client._request_api('post', endpoint, True, data=params)
        
        logger.info(f"Размещен OCO ордер: {symbol} {side}, TP: {price}, SL: {stop_price}")
        return order
    except BinanceAPIException as e:
        logger.error(f"Ошибка размещения OCO ордера: {e}")
        logger.error(f"Параметры запроса: {params}")  # Логируем параметры для диагностики
        return None
