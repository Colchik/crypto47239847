import pandas as pd
import logging
from typing import Optional, Tuple
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger("binance_utils")

def fetch_binance_klines(client: Client, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Получение исторических данных с Binance"""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        if not klines:
            logger.warning(f"Нет данных для {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Преобразование типов
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        logger.error(f"Ошибка получения данных для {symbol}: {e}")
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
    """Отмена ордера"""
    try:
        client.cancel_order(symbol=symbol, orderId=order_id)
        logger.info(f"Отменен ордер {order_id} для {symbol}")
        return True
    except BinanceAPIException as e:
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

