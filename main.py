import asyncio
import datetime
import pandas as pd
import numpy as np
import yaml
import subprocess
from binance.client import Client
from binance.exceptions import BinanceAPIException
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import nest_asyncio
import traceback
import time
import sys
import json
from collections import deque, defaultdict
import warnings

from calculations import (
    Position, TradeStats, MarketConditions, MarketRegimeDetector, 
    StopLossOptimizer, CorrelationAnalyzer, calculate_enhanced_indicators, 
    MarketMicrostructureAnalyzer
)

from ml_models import (
    EnsembleModel, AdaptiveThresholdManager, EnhancedSignalAnalyzer,
    train_enhanced_model, predict_enhanced_direction, Backtester
)

from binance_utils import (
    fetch_binance_klines, get_usdt_balance, get_symbol_price,
    get_lot_size, place_market_order, place_stop_loss_order,
    place_take_profit_order, cancel_order, get_order_book_data, 
    get_recent_trades, place_oco_order
)

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
nest_asyncio.apply()

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot_enhanced.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CryptoBotEnhanced")

def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        raise

config = load_config()

def init_clients():
    try:
        binance_client = Client(
            config['binance']['api_key'],
            config['binance']['api_secret'],
            testnet=config['binance']['testnet']
        )
        server_time = binance_client.get_server_time()
        logger.info(f"Соединение с Binance установлено. Время сервера: {datetime.datetime.fromtimestamp(server_time['serverTime']/1000)}")
        return binance_client
    except Exception as e:
        logger.error(f"Ошибка инициализации клиентов: {e}")
        raise

client = init_clients()

SYMBOLS = config['trading']['symbols']
TIMEFRAME = config['trading']['timeframe']
RISK_USD = config['trading']['risk_usd']
COMMISSION = config['trading']['commission']
SPREAD = config['trading']['spread']
HISTORICAL_DAYS = config['trading']['historical_days']
UPDATE_INTERVAL_SEC = int(config['trading']['update_interval_sec'])

class EnhancedPerformanceMonitor:
    def __init__(self):
        self.daily_stats = {
            'trades': 0,
            'profit': 0.0,
            'start_balance': 0.0,
            'max_balance': 0.0,
            'min_balance': float('inf'),
            'last_reset': datetime.datetime.now()
        }
        self.trade_history = []
        self.performance_metrics = TradeStats()
        self.symbol_stats = defaultdict(lambda: {'trades': 0, 'profit': 0, 'wins': 0})
        self.hourly_stats = defaultdict(lambda: {'trades': 0, 'profit': 0})
        self.regime_stats = defaultdict(lambda: {'trades': 0, 'profit': 0, 'wins': 0})
        self.duration_stats = {
            'very_short': {'trades': 0, 'profit': 0, 'wins': 0},
            'short': {'trades': 0, 'profit': 0, 'wins': 0},
            'medium': {'trades': 0, 'profit': 0, 'wins': 0},
            'long': {'trades': 0, 'profit': 0, 'wins': 0}
        }
        self.intraday_stats = defaultdict(lambda: {'trades': 0, 'profit': 0})
        self.recent_trades = deque(maxlen=50)

    def update_trade_stats(self, trade_info: dict):
        try:
            self.trade_history.append(trade_info)
            self.recent_trades.append(trade_info)
            self.daily_stats['trades'] += 1
            self.daily_stats['profit'] += trade_info.get('profit', 0)
            symbol = trade_info['symbol']
            self.symbol_stats[symbol]['trades'] += 1
            self.symbol_stats[symbol]['profit'] += trade_info.get('profit', 0)
            if trade_info.get('profit', 0) > 0:
                self.symbol_stats[symbol]['wins'] += 1
            hour = trade_info['timestamp'].hour
            self.hourly_stats[hour]['trades'] += 1
            self.hourly_stats[hour]['profit'] += trade_info.get('profit', 0)
            regime = trade_info.get('market_regime', 'unknown')
            self.regime_stats[regime]['trades'] += 1
            self.regime_stats[regime]['profit'] += trade_info.get('profit', 0)
            if trade_info.get('profit', 0) > 0:
                self.regime_stats[regime]['wins'] += 1
            duration_minutes = trade_info.get('duration', 0)
            if duration_minutes < 15:
                category = 'very_short'
            elif duration_minutes < 60:
                category = 'short'
            elif duration_minutes < 240:
                category = 'medium'
            else:
                category = 'long'
            self.duration_stats[category]['trades'] += 1
            self.duration_stats[category]['profit'] += trade_info.get('profit', 0)
            if trade_info.get('profit', 0) > 0:
                self.duration_stats[category]['wins'] += 1
            intraday_hour = trade_info['timestamp'].hour
            self.intraday_stats[intraday_hour]['trades'] += 1
            self.intraday_stats[intraday_hour]['profit'] += trade_info.get('profit', 0)
            self.calculate_performance_metrics()
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
            logger.error(traceback.format_exc())

    def calculate_performance_metrics(self):
        if not self.trade_history:
            return
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t['profit'] > 0]
        losing_trades = [t for t in self.trade_history if t['profit'] < 0]
        self.performance_metrics.total_trades = total_trades
        self.performance_metrics.winning_trades = len(winning_trades)
        self.performance_metrics.losing_trades = len(losing_trades)
        if total_trades > 0:
            self.performance_metrics.win_rate = len(winning_trades) / total_trades
        if winning_trades:
            self.performance_metrics.avg_win = np.mean([t['profit'] for t in winning_trades])
            total_wins = sum(t['profit'] for t in winning_trades)
        else:
            self.performance_metrics.avg_win = 0
            total_wins = 0
        if losing_trades:
            self.performance_metrics.avg_loss = np.mean([t['profit'] for t in losing_trades])
            total_losses = abs(sum(t['profit'] for t in losing_trades))
        else:
            self.performance_metrics.avg_loss = 0
            total_losses = 0
        self.performance_metrics.total_profit = sum(t['profit'] for t in self.trade_history)
        self.performance_metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        if len(self.trade_history) > 1:
            returns = [t['profit_percent'] for t in self.trade_history if 'profit_percent' in t]
            if returns:
                self.performance_metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        cumulative_profits = np.cumsum([t['profit'] for t in self.trade_history])
        running_max = np.maximum.accumulate(cumulative_profits)
        drawdowns = running_max - cumulative_profits
        self.performance_metrics.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        max_wins = max_losses = current_wins = current_losses = 0
        for trade in self.trade_history:
            if trade['profit'] > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        self.performance_metrics.max_consecutive_wins = max_wins
        self.performance_metrics.max_consecutive_losses = max_losses

    def get_best_trading_hours(self):
        best_hours = []
        for hour, stats in self.hourly_stats.items():
            if stats['trades'] > 0:
                best_hours.append((hour, {
                    'avg_profit': stats['profit'] / stats['trades'],
                    'trades': stats['trades'],
                    'profit': stats['profit']
                }))
        return sorted(best_hours, key=lambda x: x[1]['avg_profit'], reverse=True)

    def get_best_symbols(self, limit: int = 10):
        symbol_list = []
        for symbol, stats in self.symbol_stats.items():
            if stats['trades'] > 0:
                symbol_list.append((symbol, {
                    'trades': stats['trades'],
                    'profit': stats['profit'],
                    'win_rate': stats['wins'] / stats['trades'],
                    'avg_profit': stats['profit'] / stats['trades']
                }))
        return sorted(symbol_list, key=lambda x: x[1]['profit'], reverse=True)[:limit]

    def get_regime_performance(self):
        regime_perf = {}
        for regime, stats in self.regime_stats.items():
            if stats['trades'] > 0:
                regime_perf[regime] = {
                    'trades': stats['trades'],
                    'profit': stats['profit'],
                    'win_rate': stats['wins'] / stats['trades'],
                    'avg_profit': stats['profit'] / stats['trades']
                }
        return regime_perf

    def get_duration_performance(self) -> Dict[str, dict]:
        duration_performance = {}
        for category, stats in self.duration_stats.items():
            if stats['trades'] > 0:
                duration_performance[category] = {
                    'trades': stats['trades'],
                    'profit': stats['profit'],
                    'win_rate': stats['wins'] / stats['trades'],
                    'avg_profit': stats['profit'] / stats['trades']
                }
        return duration_performance

    def get_intraday_performance(self) -> List[Tuple[int, dict]]:
        intraday_performance = []
        for hour, stats in self.intraday_stats.items():
            if stats['trades'] > 0:
                intraday_performance.append((
                    hour,
                    {
                        'trades': stats['trades'],
                        'profit': stats['profit'],
                        'avg_profit': stats['profit'] / stats['trades']
                    }
                ))
        return sorted(intraday_performance, key=lambda x: x[0])

    def get_recent_performance(self) -> dict:
        if not self.recent_trades:
            return {
                'trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0
            }
        recent_profits = [t.get('profit', 0) for t in self.recent_trades]
        winning_trades = [p for p in recent_profits if p > 0]
        return {
            'trades': len(recent_profits),
            'win_rate': len(winning_trades) / len(recent_profits) if recent_profits else 0,
            'avg_profit': np.mean(recent_profits) if recent_profits else 0,
            'total_profit': sum(recent_profits)
        }

    def reset_daily_stats(self):
        self.daily_stats = {
            'trades': 0,
            'profit': 0.0,
            'start_balance': get_usdt_balance(client) or 0.0,
            'max_balance': 0.0,
            'min_balance': float('inf'),
            'last_reset': datetime.datetime.now()
        }

class EnhancedPositionManager:
    def __init__(self, client, risk_percent=0.01, max_positions=8, max_correlation=0.8):
        self.client = client
        self.risk_percent = risk_percent
        self.max_positions = max_positions
        self.max_correlation = max_correlation
        self.positions: Dict[str, Position] = {}
        self.stop_loss_orders: Dict[str, str] = {}
        self.take_profit_orders: Dict[str, str] = {}
        self.position_history = []
        self.correlation_analyzer = CorrelationAnalyzer(SYMBOLS)
        self.stop_loss_optimizer = StopLossOptimizer()
        self.microstructure_analyzer = MarketMicrostructureAnalyzer()
        self.max_position_duration = 24 * 60 * 60

    def _get_performance_multiplier(self) -> float:
        if len(self.position_history) < 5:
            return 1.0
        recent_trades = self.position_history[-20:]
        winning_trades = [t for t in recent_trades if t.get('profit', 0) > 0]
        win_rate = len(winning_trades) / len(recent_trades)
        avg_profit_percent = np.mean([t.get('profit_percent', 0) for t in recent_trades])
        if win_rate > 0.6 and avg_profit_percent > 0.5:
            return 1.2
        elif win_rate < 0.4 or avg_profit_percent < -0.5:
            return 0.8
        else:
            return 1.0

    def can_open_position(self, symbol: str) -> bool:
        if len(self.positions) >= self.max_positions:
            logger.info(f"Достигнуто максимальное количество позиций: {self.max_positions}")
            return False
        if symbol in self.positions:
            logger.info(f"Позиция для {symbol} уже открыта")
            return False
        if not self.correlation_analyzer.should_trade_symbol(symbol, self.positions, self.max_correlation):
            logger.info(f"Высокая корреляция {symbol} с открытыми позициями")
            return False
        return True

    def calculate_dynamic_position_size(self, symbol: str, price: float, atr: float,
                                      market_conditions: MarketConditions,
                                      confidence_score: float,
                                      microstructure_data: dict = None) -> Tuple[float, float]:
        try:
            balance = get_usdt_balance(self.client)
            if balance is None or balance <= 0:
                logger.warning(f"Недостаточный баланс: {balance}")
                return 0.0, 0.0
            base_risk = self.risk_percent * 1.2
            risk_multipliers = {
                "high_volatility": 0.6,
                "trending_up": 1.3,
                "trending_down": 1.3,
                "ranging": 0.8,
                "unknown": 0.8
            }
            market_multiplier = risk_multipliers.get(market_conditions.regime, 1.0)
            volatility_multiplier = 1.0
            if market_conditions.volatility_percentile > 80:
                volatility_multiplier = 0.7
            elif market_conditions.volatility_percentile < 20:
                volatility_multiplier = 1.3
            confidence_multiplier = 0.5 + confidence_score
            performance_multiplier = self._get_performance_multiplier()
            microstructure_multiplier = 1.0
            if microstructure_data:
                imbalance = microstructure_data.get('imbalance', 0)
                if abs(imbalance) > 0.3:
                    microstructure_multiplier = 1.2
                if microstructure_data.get('price_jump', False):
                    microstructure_multiplier = 0.7
                if microstructure_data.get('spread_trend', 0) > 0.01:
                    microstructure_multiplier *= 0.8
            adjusted_risk = base_risk * market_multiplier * volatility_multiplier * \
                          confidence_multiplier * performance_multiplier * microstructure_multiplier
            adjusted_risk = np.clip(adjusted_risk, 0.005, 0.04)
            risk_amount = balance * adjusted_risk
            stop_distance_multiplier = self.stop_loss_optimizer.atr_multipliers.get(market_conditions.regime, 2.0) * 0.8
            stop_distance = atr * stop_distance_multiplier
            if microstructure_data and 'market_impact' in microstructure_data:
                market_impact = microstructure_data['market_impact']
                if market_impact > 0.002:
                    risk_amount *= (1 - market_impact * 10)
            position_size_usd = risk_amount
            position_size = position_size_usd / price
            min_qty, qty_precision = get_lot_size(self.client, symbol)
            position_size = max(round(position_size, qty_precision), min_qty)
            max_position = balance * 0.12 / price
            position_size = min(position_size, max_position)
            logger.info(f"Рассчитан размер позиции для {symbol}: {position_size} "
                       f"(${position_size*price:.2f}), риск: {adjusted_risk*100:.2f}%")
            return position_size, stop_distance
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            logger.error(traceback.format_exc())
            return 0.0, 0.0

    async def open_position(self, symbol: str, side: str, entry_price: float,
                          quantity: float, stop_loss: float, take_profit: float,
                          market_conditions: MarketConditions, confidence_score: float) -> Optional[Position]:
        try:
            order = place_market_order(self.client, symbol, "BUY" if side == "LONG" else "SELL", quantity)
            if not order:
                logger.error(f"Не удалось открыть позицию для {symbol}")
                return None
            position = Position(
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity,
                side=side,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=stop_loss,
                trailing_take_profit=take_profit,
                entry_time=datetime.datetime.now(),
                order_id=order.get('orderId'),
                market_regime=market_conditions.regime,
                confidence_score=confidence_score,
                max_profit=0.0
            )
            self.positions[symbol] = position
            await self.place_sl_tp_orders(position)
            logger.info(f"Открыта позиция: {symbol} {side}, количество: {quantity}, "
                       f"цена: {entry_price}, SL: {stop_loss}, TP: {take_profit}")
            return position
        except Exception as e:
            logger.error(f"Ошибка открытия позиции: {e}")
            logger.error(traceback.format_exc())
            return None

    async def place_sl_tp_orders(self, position: Position):
        try:
            # Определяем сторону для закрытия позиции
            close_side = "SELL" if position.side == "LONG" else "BUY"
            
            # Отменяем существующие ордера, если есть
            if position.symbol in self.stop_loss_orders:
                cancel_order(self.client, position.symbol, self.stop_loss_orders[position.symbol])
                del self.stop_loss_orders[position.symbol]
                
            if position.symbol in self.take_profit_orders:
                cancel_order(self.client, position.symbol, self.take_profit_orders[position.symbol])
                del self.take_profit_orders[position.symbol]
            
            # Размещаем OCO ордер вместо отдельных SL и TP
            oco_order = place_oco_order(
                self.client,
                position.symbol,
                close_side,
                position.quantity,
                position.take_profit,  # Take profit цена
                position.stop_loss,    # Stop loss цена
            )
            
            if oco_order:
                # Сохраняем ID ордера - у OCO ордера может быть несколько ID
                orders = oco_order.get('orderReports', [])
                for order in orders:
                    order_type = order.get('type')
                    if order_type == 'STOP_LOSS_LIMIT':
                        self.stop_loss_orders[position.symbol] = order.get('orderId')
                    elif order_type == 'LIMIT_MAKER':
                        self.take_profit_orders[position.symbol] = order.get('orderId')
                
                logger.info(f"Размещен OCO ордер для {position.symbol}: TP={position.take_profit}, SL={position.stop_loss}")
            else:
                logger.error(f"Не удалось разместить OCO ордер для {position.symbol}")
                
        except Exception as e:
            logger.error(f"Ошибка размещения OCO ордеров: {e}")
            logger.error(traceback.format_exc())

    async def close_position(self, symbol: str, exit_price: float, reason: str):
        try:
            position = self.positions.get(symbol)
            if not position:
                logger.warning(f"Позиция {symbol} не найдена")
                return
            if hasattr(position, '_closing'):
                logger.warning(f"Позиция {symbol} уже закрывается")
                return
            position._closing = True
            balance_before = get_usdt_balance(self.client)
            logger.info(f"Баланс ДО закрытия {symbol}: {balance_before:.2f} USDT")
            if symbol in self.stop_loss_orders:
                cancel_order(self.client, symbol, self.stop_loss_orders[symbol])
                del self.stop_loss_orders[symbol]
            if symbol in self.take_profit_orders:
                cancel_order(self.client, symbol, self.take_profit_orders[symbol])
                del self.take_profit_orders[symbol]
            close_side = "SELL" if position.side == "LONG" else "BUY"
            order = place_market_order(self.client, symbol, close_side, position.quantity)
            if not order:
                logger.error(f"Не удалось закрыть позицию: {symbol}")
                del position._closing
                return
            executed_price = float(order.get('fills', [{}])[0].get('price', exit_price))
            executed_qty = float(order.get('executedQty', position.quantity))
            slippage = abs(executed_price - exit_price) / exit_price * 100
            if slippage > 0.5:
                logger.warning(f"Высокое проскальзывание при закрытии {symbol}: {slippage:.2f}%")
            if position.side == "LONG":
                gross_profit = executed_qty * (executed_price - position.entry_price)
            else:
                gross_profit = executed_qty * (position.entry_price - executed_price)
            entry_commission = position.quantity * position.entry_price * COMMISSION
            exit_commission = executed_qty * executed_price * COMMISSION
            total_commission = entry_commission + exit_commission
            net_profit = gross_profit - total_commission
            logger.info(f"""
            Закрытие {symbol} {position.side}:
            - Вход: {position.entry_price}, Выход: {executed_price}
            - Количество: {executed_qty}
            - Валовая прибыль: {gross_profit:.4f}
            - Комиссии: {total_commission:.4f}
            - Чистая прибыль: {net_profit:.4f}
            """)
            await asyncio.sleep(1)
            balance_after = get_usdt_balance(self.client)
            actual_change = balance_after - balance_before
            logger.info(f"""
            ПРОВЕРКА БАЛАНСА:
            - Баланс до: {balance_before:.2f} USDT
            - Баланс после: {balance_after:.2f} USDT
            - Фактическое изменение: {actual_change:.2f} USDT
            - Расчетная прибыль: {net_profit:.2f} USDT
            - Расхождение: {abs(actual_change - net_profit):.2f} USDT
            """)
            if abs(actual_change - net_profit) > 1.0:
                logger.warning(f"БОЛЬШОЕ РАСХОЖДЕНИЕ! Детали ордера: {json.dumps(order, indent=2)}")
            trade_record = {
                'symbol': symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': executed_price,
                'quantity': executed_qty,
                'gross_profit': gross_profit,
                'commission': total_commission,
                'profit': net_profit,
                'profit_percent': net_profit / (position.quantity * position.entry_price) * 100,
                'entry_time': position.entry_time,
                'exit_time': datetime.datetime.now(),
                'duration': (datetime.datetime.now() - position.entry_time).total_seconds() / 60,
                'market_regime': position.market_regime,
                'confidence_score': position.confidence_score,
                'reason': reason,
                'timestamp': datetime.datetime.now(),
                'actual_balance_change': actual_change,
                'balance_discrepancy': abs(actual_change - net_profit)
            }
            self.position_history.append(trade_record)
            del self.positions[symbol]
            logger.info(f"Закрыта позиция: {symbol} {position.side}, прибыль: {net_profit:.2f} USDT ({trade_record['profit_percent']:.2f}%), причина: {reason}")
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции: {e}")
            logger.error(traceback.format_exc())
            if position and hasattr(position, '_closing'):
                del position._closing

    async def _update_stop_loss(self, symbol: str, position: Position, new_stop: float):
        try:
            if symbol in self.stop_loss_orders:
                cancel_order(self.client, symbol, self.stop_loss_orders[symbol])
            sl_side = "SELL" if position.side == "LONG" else "BUY"
            sl_order = place_stop_loss_order(
                self.client, symbol, sl_side,
                position.quantity, new_stop
            )
            if sl_order:
                self.stop_loss_orders[symbol] = sl_order.get('orderId')
                position.trailing_stop = new_stop
                logger.debug(f"Обновлен стоп-лосс для {symbol}: {new_stop}")
        except Exception as e:
            logger.error(f"Ошибка обновления стоп-лосса: {e}")

    async def _update_take_profit(self, symbol: str, position: Position, new_tp: float):
        try:
            if symbol in self.take_profit_orders:
                cancel_order(self.client, symbol, self.take_profit_orders[symbol])
            tp_side = "SELL" if position.side == "LONG" else "BUY"
            tp_order = place_take_profit_order(
                self.client, symbol, tp_side,
                position.quantity, new_tp
            )
            if tp_order:
                self.take_profit_orders[symbol] = tp_order.get('orderId')
                position.trailing_take_profit = new_tp
                logger.debug(f"Обновлен тейк-профит для {symbol}: {new_tp}")
        except Exception as e:
            logger.error(f"Ошибка обновления тейк-профита: {e}")

    async def update_trailing_stops(self, current_prices: Dict[str, float], microstructure_data: Dict[str, dict] = None):
        """Обновление трейлинг-стопов с использованием OCO ордеров"""
        for symbol, position in list(self.positions.items()):
            try:
                if symbol not in current_prices:
                    continue
                    
                # Проверка существования позиции на бирже
                # Если позиция уже закрылась через OCO ордер, нужно обновить локальное состояние
                try:
                    # Получаем открытые ордера для символа
                    open_orders = self.client.get_open_orders(symbol=symbol)
                    
                    # Если у нас есть ордера в стоп-списке или тейк-профит списке, но их нет на бирже,
                    # и при этом мы не видим новых транзакций - вероятно, ордер сработал
                    sl_order_id = self.stop_loss_orders.get(symbol)
                    tp_order_id = self.take_profit_orders.get(symbol)
                    
                    if (sl_order_id or tp_order_id) and not any(o.get('orderId') == sl_order_id or o.get('orderId') == tp_order_id for o in open_orders):
                        # Проверим, изменился ли баланс - признак того, что ордер исполнился
                        balance_now = get_usdt_balance(self.client)
                        # Простая проверка - если баланс изменился, вероятно позиция закрылась
                        if balance_now != getattr(self, '_last_balance', None):
                            logger.info(f"Обнаружено автоматическое закрытие позиции {symbol} через OCO ордер")
                            # Обновляем локальное состояние
                            self._last_balance = balance_now
                            if symbol in self.positions:
                                del self.positions[symbol]
                            if symbol in self.stop_loss_orders:
                                del self.stop_loss_orders[symbol]
                            if symbol in self.take_profit_orders:
                                del self.take_profit_orders[symbol]
                            continue
                except Exception as e:
                    logger.warning(f"Ошибка проверки статуса ордеров для {symbol}: {e}")
                
                current_price = current_prices[symbol]
                current_time = datetime.datetime.now()
                position_duration = (current_time - position.entry_time).total_seconds()
                
                # Проверка максимальной длительности позиции
                if position_duration > self.max_position_duration:
                    logger.info(f"Позиция {symbol} достигла максимальной длительности ({position_duration/3600:.1f} ч)")
                    await self.close_position(symbol, current_price, "Максимальная длительность")
                    continue
                
                # Расчет текущей прибыли
                if position.side == "LONG":
                    current_profit = (current_price - position.entry_price) / position.entry_price
                else:
                    current_profit = (position.entry_price - current_price) / position.entry_price
                    
                position.max_profit = max(position.max_profit, current_profit)
                
                # Проверка срабатывания стопов (на случай, если OCO не сработал)
                if position.side == "LONG":
                    if current_price <= position.trailing_stop or current_price >= position.trailing_take_profit:
                        await self.close_position(symbol, current_price, "SL/TP triggered (manual check)")
                        continue
                elif position.side == "SHORT":
                    if current_price >= position.trailing_stop or current_price <= position.trailing_take_profit:
                        await self.close_position(symbol, current_price, "SL/TP triggered (manual check)")
                        continue
                
                # Логика трейлинг-стопа
                update_needed = False
                new_stop = position.trailing_stop
                new_tp = position.trailing_take_profit
                
                if position.side == "LONG":
                    trailing_distance = current_price * 0.015
                    potential_new_stop = current_price - trailing_distance
                    
                    # Корректировка стопа на основе микроструктуры рынка
                    if microstructure_data and symbol in microstructure_data:
                        micro_data = microstructure_data[symbol]
                        if micro_data.get('pressure_ratio', 1) < 0.8:
                            potential_new_stop = current_price - trailing_distance * 0.7
                        if micro_data.get('price_jump', False):
                            potential_new_stop = current_price - trailing_distance * 0.5
                    
                    # Только если новый стоп выше текущего - перемещаем его вверх
                    if potential_new_stop > position.trailing_stop:
                        new_stop = potential_new_stop
                        update_needed = True
                        
                    # Обновление тейк-профита, если позиция в прибыли
                    if current_profit > 0.4 * ((position.take_profit - position.entry_price) / position.entry_price):
                        potential_new_tp = current_price + (current_price - new_stop) * 1.2
                        if potential_new_tp < position.trailing_take_profit:
                            new_tp = potential_new_tp
                            update_needed = True
                            
                elif position.side == "SHORT":
                    trailing_distance = current_price * 0.015
                    potential_new_stop = current_price + trailing_distance
                    
                    # Корректировка стопа на основе микроструктуры рынка
                    if microstructure_data and symbol in microstructure_data:
                        micro_data = microstructure_data[symbol]
                        if micro_data.get('pressure_ratio', 1) > 1.2:
                            potential_new_stop = current_price + trailing_distance * 0.7
                        if micro_data.get('price_jump', False):
                            potential_new_stop = current_price + trailing_distance * 0.5
                    
                    # Только если новый стоп ниже текущего - перемещаем его вниз
                    if potential_new_stop < position.trailing_stop:
                        new_stop = potential_new_stop
                        update_needed = True
                        
                    # Обновление тейк-профита, если позиция в прибыли
                    if current_profit > 0.4 * ((position.entry_price - position.take_profit) / position.entry_price):
                        potential_new_tp = current_price - (new_stop - current_price) * 1.2
                        if potential_new_tp > position.trailing_take_profit:
                            new_tp = potential_new_tp
                            update_needed = True
                
                # Если нужно обновить ордера - отменяем старые и создаем новый OCO
                if update_needed:
                    try:
                        # Отмена существующих ордеров
                        if symbol in self.stop_loss_orders:
                            cancel_order(self.client, symbol, self.stop_loss_orders[symbol])
                            del self.stop_loss_orders[symbol]
                            
                        if symbol in self.take_profit_orders:
                            cancel_order(self.client, symbol, self.take_profit_orders[symbol])
                            del self.take_profit_orders[symbol]
                            
                        # Создание нового OCO ордера
                        close_side = "SELL" if position.side == "LONG" else "BUY"
                        oco_order = place_oco_order(
                            self.client,
                            symbol,
                            close_side,
                            position.quantity,
                            new_tp,  # Take profit
                            new_stop  # Stop loss
                        )
                        
                        if oco_order:
                            # Сохраняем ID ордера - у OCO может быть несколько ID
                            orders = oco_order.get('orderReports', [])
                            for order in orders:
                                order_type = order.get('type')
                                if order_type == 'STOP_LOSS_LIMIT':
                                    self.stop_loss_orders[symbol] = order.get('orderId')
                                elif order_type == 'LIMIT_MAKER':
                                    self.take_profit_orders[symbol] = order.get('orderId')
                                    
                            # Обновляем значения в позиции
                            position.trailing_stop = new_stop
                            position.trailing_take_profit = new_tp
                            logger.info(f"Обновлен OCO ордер для {symbol}: SL={new_stop}, TP={new_tp}")
                    except Exception as e:
                        logger.error(f"Ошибка обновления OCO ордера для {symbol}: {e}")
                
                # Частичное закрытие позиции в прибыли
                if position.max_profit > 0.02 and not hasattr(position, 'partially_closed'):
                    if current_profit > 0.015:
                        partial_quantity = position.quantity * 0.5
                        await self._partial_close(symbol, position, partial_quantity, current_price, "Частичная фиксация")
                        position.partially_closed = True
            
            except Exception as e:
                logger.error(f"Ошибка обработки трейлинг-стопа для {symbol}: {e}")
                logger.error(traceback.format_exc())

    async def _partial_close(self, symbol: str, position: Position, quantity: float, current_price: float, reason: str):
        try:
            close_side = "SELL" if position.side == "LONG" else "BUY"
            
            # Отменяем существующие OCO ордера
            if symbol in self.stop_loss_orders:
                cancel_order(self.client, symbol, self.stop_loss_orders[symbol])
                del self.stop_loss_orders[symbol]
                
            if symbol in self.take_profit_orders:
                cancel_order(self.client, symbol, self.take_profit_orders[symbol])
                del self.take_profit_orders[symbol]
            
            # Размещаем рыночный ордер на частичное закрытие
            order = place_market_order(self.client, symbol, close_side, quantity)
            if not order:
                logger.error(f"Не удалось частично закрыть позицию: {symbol}")
                return
                
            # Расчет прибыли
            if position.side == "LONG":
                profit = quantity * (current_price - position.entry_price)
            else:
                profit = quantity * (position.entry_price - current_price)
                
            profit -= quantity * current_price * COMMISSION * 2
            
            # Запись в историю позиций
            trade_record = {
                'symbol': symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'quantity': quantity,
                'profit': profit,
                'profit_percent': profit / (quantity * position.entry_price) * 100,
                'entry_time': position.entry_time,
                'exit_time': datetime.datetime.now(),
                'duration': (datetime.datetime.now() - position.entry_time).total_seconds() / 60,
                'market_regime': position.market_regime,
                'confidence_score': position.confidence_score,
                'max_profit': position.max_profit,
                'reason': reason + " (частично)",
                'timestamp': datetime.datetime.now()
            }
            self.position_history.append(trade_record)
            
            # Обновляем позицию
            position.quantity -= quantity
            
            # Создаем новый OCO ордер для оставшейся части позиции
            await self.place_sl_tp_orders(position)
            
            logger.info(f"Частично закрыта позиция: {symbol} {position.side}, {quantity} из {position.quantity + quantity}, "
                    f"прибыль {profit:.2f} USDT ({trade_record['profit_percent']:.2f}%)")
                    
        except Exception as e:
            logger.error(f"Ошибка частичного закрытия позиции: {e}")
            logger.error(traceback.format_exc())

class EnhancedTradingEngine:
    def __init__(self, client, position_manager: EnhancedPositionManager,
                 signal_analyzer: EnhancedSignalAnalyzer, monitor: EnhancedPerformanceMonitor):
        self.client = client
        self.position_manager = position_manager
        self.signal_analyzer = signal_analyzer
        self.monitor = monitor
        self.models = {}
        self.is_running = False
        self._lock = asyncio.Lock()
        self.last_model_update = datetime.datetime.now() - datetime.timedelta(days=1)
        self.market_detector = MarketRegimeDetector()
        self.threshold_manager = AdaptiveThresholdManager()
        self.commission = COMMISSION
        self.notification_manager = None

    async def process_symbol(self, symbol: str, microstructure_data: dict = None):
        try:
            model_data = self.models.get(symbol)
            if not model_data:
                logger.warning(f"[process_symbol] Нет модели для {symbol}")
                return
            if len(model_data) == 6:
                model, scaler, features, threshold, performance, all_features = model_data
            else:
                model, scaler, features, threshold, performance = model_data
                all_features = None
            df = fetch_binance_klines(self.client, symbol, TIMEFRAME, limit=200)
            if df.empty or len(df) < 50:
                logger.warning(f"[process_symbol] Недостаточно данных для {symbol}")
                return
            df = calculate_enhanced_indicators(df)
            if df.empty:
                logger.warning(f"[process_symbol] Не удалось рассчитать индикаторы для {symbol}")
                return
            market_conditions = self.market_detector.detect_regime(df)
            label, prob = predict_enhanced_direction(
                df, model, scaler, features, self.commission, threshold, all_features
            )
            confidence_score = abs(prob - 0.5) * 2
            signal, strength, reason, final_confidence = self.signal_analyzer.analyze_signal(
                df, prob, market_conditions, confidence_score, microstructure_data
            )
            logger.info(f"[{symbol}] Сигнал: {signal}, Уверенность: {final_confidence:.2f}, Причина: {reason}")
            current_price = df['close'].iloc[-1]
            if signal in ["BUY", "SELL"] and self.position_manager.can_open_position(symbol):
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.01
                side = "LONG" if signal == "BUY" else "SHORT"
                size, sl_distance = self.position_manager.calculate_dynamic_position_size(
                    symbol, current_price, atr, market_conditions, confidence_score, microstructure_data
                )
                if size <= 0 or sl_distance <= 0:
                    logger.warning(f"[{symbol}] Нулевой размер позиции/стопа")
                    return
                if side == "LONG":
                    stop_loss = current_price - sl_distance
                    take_profit = current_price + sl_distance * 1.5
                else:
                    stop_loss = current_price + sl_distance
                    take_profit = current_price - sl_distance * 1.5
                min_stop_distance = current_price * 0.005
                if side == "LONG" and (current_price - stop_loss) < min_stop_distance:
                    stop_loss = current_price - min_stop_distance
                elif side == "SHORT" and (stop_loss - current_price) < min_stop_distance:
                    stop_loss = current_price + min_stop_distance
                position = await self.position_manager.open_position(
                    symbol, side, current_price, size,
                    stop_loss, take_profit, market_conditions, confidence_score
                )
                if position:
                    self.monitor.update_trade_stats({
                        'symbol': symbol,
                        'side': side,
                        'entry_price': current_price,
                        'quantity': size,
                        'timestamp': datetime.datetime.now(),
                        'market_regime': market_conditions.regime,
                        'confidence_score': confidence_score,
                        'profit': 0
                    })
                    if self.notification_manager:
                        await self.notification_manager.send_trade_notification(position, "OPEN")
        except Exception as e:
            logger.error(f"[process_symbol] Ошибка обработки {symbol}: {e}")
            logger.error(traceback.format_exc())

    def _simple_backtest(self, df, model, scaler, features, threshold, all_features=None):
        try:
            logger.info("Запуск упрощенного бэктеста")
            min_lookback = 100
            if len(df) < min_lookback + 50:
                logger.warning("Недостаточно данных для бэктеста")
                return {'win_rate': 0.5, 'profit_factor': 1.0, 'total_trades': 0}
            trades = []
            for i in range(min_lookback, min(len(df), min_lookback + 500)):
                if i % 50 == 0:
                    logger.debug(f"Бэктест прогресс: {i}/{min(len(df), min_lookback + 500)}")
                current_data = df.iloc[:i]
                try:
                    label, prob_up = predict_enhanced_direction(
                        current_data, model, scaler, features, self.commission, threshold, all_features
                    )
                    if prob_up > threshold:
                        entry_price = df.iloc[i]['close']
                        for j in range(1, min(20, len(df) - i)):
                            exit_price = df.iloc[i + j]['close']
                            profit = (exit_price - entry_price) / entry_price
                            if profit < -0.02:
                                trades.append(profit - self.commission * 2)
                                break
                            elif profit > 0.03:
                                trades.append(profit - self.commission * 2)
                                break
                            elif j == 19:
                                trades.append(profit - self.commission * 2)
                    elif prob_up < 1 - threshold:
                        entry_price = df.iloc[i]['close']
                        for j in range(1, min(20, len(df) - i)):
                            exit_price = df.iloc[i + j]['close']
                            profit = (entry_price - exit_price) / entry_price
                            if profit < -0.02:
                                trades.append(profit - self.commission * 2)
                                break
                            elif profit > 0.03:
                                trades.append(profit - self.commission * 2)
                                break
                            elif j == 19:
                                trades.append(profit - self.commission * 2)
                except Exception as e:
                    logger.debug(f"Ошибка предсказания в бэктесте: {e}")
                    continue
            logger.info(f"Упрощенный бэктест завершен, сделок: {len(trades)}")
            if trades:
                winning_trades = [t for t in trades if t > 0]
                losing_trades = [t for t in trades if t < 0]
                win_rate = len(winning_trades) / len(trades)
                total_wins = sum(winning_trades) if winning_trades else 0
                total_losses = abs(sum(losing_trades)) if losing_trades else 0.0001
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                if len(trades) > 1:
                    returns = np.array(trades)
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                else:
                    sharpe = 0
                cumulative_returns = np.cumprod(1 + np.array(trades)) - 1
                running_max = np.maximum.accumulate(cumulative_returns + 1)
                drawdown = (running_max - (cumulative_returns + 1)) / running_max
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
                max_consecutive_wins = 0
                max_consecutive_losses = 0
                current_wins = 0
                current_losses = 0
                for trade in trades:
                    if trade > 0:
                        current_wins += 1
                        current_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, current_wins)
                    else:
                        current_losses += 1
                        current_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, current_losses)
                total_return = np.prod(1 + np.array(trades)) - 1
                logger.info(f"Детали бэктеста: Win Rate: {win_rate:.2%}, Profit Factor: {profit_factor:.2f}, "
                           f"Avg Win: {avg_win:.4f}, Avg Loss: {avg_loss:.4f}, Sharpe: {sharpe:.2f}, "
                           f"Max DD: {max_drawdown:.2%}, Total Return: {total_return:.2%}")
                return {
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_trades': len(trades),
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'total_return': total_return,
                    'max_consecutive_wins': max_consecutive_wins,
                    'max_consecutive_losses': max_consecutive_losses,
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades)
                }
            else:
                return {
                    'win_rate': 0.5,
                    'profit_factor': 1.0,
                    'total_trades': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_return': 0,
                    'max_consecutive_wins': 0,
                    'max_consecutive_losses': 0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }
        except Exception as e:
            logger.error(f"Ошибка в упрощенном бэктесте: {e}")
            logger.error(traceback.format_exc())
            return {
                'win_rate': 0.5,
                'profit_factor': 1.0,
                'total_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }

    async def optimize_model_parameters(self, symbol: str):
        try:
            if symbol not in self.models:
                logger.error(f"Модель для {symbol} не найдена")
                return False
            model_data = self.models[symbol]
            if len(model_data) == 6:
                model, scaler, features, threshold, _, all_features = model_data
            else:
                model, scaler, features, threshold, _ = model_data
                all_features = None
            df = fetch_binance_klines(self.client, symbol, TIMEFRAME, 1000)
            if df.empty:
                logger.error(f"Не удалось получить данные для {symbol}")
                return False
            logger.info(f"Запуск оптимизации параметров для {symbol}")
            thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
            best_threshold = threshold
            best_score = 0
            best_results = None
            for test_threshold in thresholds:
                backtester = Backtester()
                results = backtester.backtest_strategy(
                    df, model, scaler, features,
                    self.signal_analyzer,
                    self.market_detector,
                    self.threshold_manager,
                    test_threshold,
                    all_features
                )
                if results['total_trades'] < 10:
                    continue
                score = (
                    results['win_rate'] * 0.4 +
                    min(results['profit_factor'], 3.0) / 3.0 * 0.4 +
                    min(results['total_trades'], 100) / 100 * 0.2
                )
                logger.info(f"Порог {test_threshold:.2f}: WR={results['win_rate']:.2%}, "
                           f"PF={results['profit_factor']:.2f}, Сделок={results['total_trades']}, "
                           f"Оценка={score:.2f}")
                if score > best_score:
                    best_score = score
                    best_threshold = test_threshold
                    best_results = results
            if best_results and best_threshold != threshold:
                logger.info(f"Оптимизирован порог для {symbol}: {threshold:.2f} -> {best_threshold:.2f}")
                self.models[symbol] = (model, scaler, features, best_threshold, best_results, all_features)
                return True
            else:
                logger.info(f"Порог для {symbol} остался прежним: {threshold:.2f}")
                return False
        except Exception as e:
            logger.error(f"Ошибка оптимизации параметров для {symbol}: {e}")
            logger.error(traceback.format_exc())
            return False

    async def initialize_models(self):
        async with self._lock:
            try:
                for symbol in SYMBOLS:
                    logger.info(f"Загрузка исторических данных для {symbol}")
                    
                    # Загружаем данные за 14 дней
                    df = fetch_binance_klines(self.client, symbol, TIMEFRAME, days=14)
                    
                    logger.info(f"Загружено {len(df)} строк данных для {symbol}")
                    
                    if df.empty or len(df) < 100:
                        logger.error(f"Недостаточно данных для {symbol} (загружено: {len(df)} строк)")
                        continue
                            
                    logger.info(f"Обучение модели для {symbol}")
                    result = train_enhanced_model(df, self.commission)
                    if len(result) == 5:
                        model, scaler, features, threshold, all_features = result
                    else:
                        model, scaler, features, threshold = result
                        all_features = None
                    if model is None or scaler is None or not features:
                        logger.error(f"Не удалось обучить модель для {symbol}")
                        continue
                    logger.info(f"Запуск бэктеста для {symbol}")
                    try:
                        backtester = Backtester()
                        backtest_results = backtester.backtest_strategy(
                            df, model, scaler, features,
                            self.signal_analyzer,
                            self.market_detector,
                            self.threshold_manager,
                            threshold,
                            all_features
                        )
                        logger.info(f"Результаты бэктеста для {symbol}: "
                                f"WR={backtest_results['win_rate']:.2%}, "
                                f"PF={backtest_results['profit_factor']:.2f}, "
                                f"Сделок={backtest_results['total_trades']}")
                        self.models[symbol] = (model, scaler, features, threshold, backtest_results, all_features)
                        if backtest_results['win_rate'] < 0.45:
                            new_threshold = min(threshold + 0.05, 0.65)
                            logger.info(f"Адаптация порога для {symbol}: {threshold:.2f} -> {new_threshold:.2f}")
                            self.models[symbol] = (model, scaler, features, new_threshold, backtest_results, all_features)
                        logger.info(f"Модель для {symbol} добавлена")
                    except Exception as e:
                        logger.error(f"Ошибка бэктеста для {symbol}: {e}")
                        logger.error(traceback.format_exc())
                        self.models[symbol] = (model, scaler, features, threshold, {
                            'win_rate': 0.5,
                            'profit_factor': 1.0,
                            'total_trades': 0
                        }, all_features)
                if self.models:
                    logger.info(f"Успешно обучены модели для {len(self.models)} символов: {list(self.models.keys())}")
                    self.last_model_update = datetime.datetime.now()
                    try:
                        logger.info("Обновление корреляций...")
                        self.position_manager.correlation_analyzer.update_correlations(self.client)
                        logger.info("Корреляции обновлены")
                    except Exception as e:
                        logger.error(f"Ошибка обновления корреляций: {e}")
                else:
                    logger.error("Не удалось обучить ни одной модели")
            except Exception as e:
                logger.error(f"Ошибка инициализации моделей: {e}")
                logger.error(traceback.format_exc())
                raise


class EnhancedTelegramBot:
    def __init__(self):
        self.trading_engine = None
        self.application = None
        self.notification_manager = None
        self.trading_task = None

    async def setup(self):
        try:
            position_manager = EnhancedPositionManager(client)
            signal_analyzer = EnhancedSignalAnalyzer(COMMISSION, SPREAD)
            monitor = EnhancedPerformanceMonitor()
            self.trading_engine = EnhancedTradingEngine(client, position_manager, signal_analyzer, monitor)
            self.application = Application.builder().token(config['telegram']['token']).build()
            self.notification_manager = EnhancedNotificationManager(
                self.application.bot,
                config['telegram']['chat_id']
            )
            self.trading_engine.notification_manager = self.notification_manager
            telegram_handler = EnhancedTelegramHandler(self.trading_engine, monitor)
            self.application.add_handler(CommandHandler("start", telegram_handler.handle_start))
            self.application.add_handler(CommandHandler("status", telegram_handler.handle_status))
            self.application.add_handler(CommandHandler("stats", telegram_handler.handle_stats))
            self.application.add_handler(CommandHandler("positions", telegram_handler.handle_positions))
            self.application.add_handler(CommandHandler("history", telegram_handler.handle_history))
            self.application.add_handler(CommandHandler("start_trading", telegram_handler.handle_start_trading))
            self.application.add_handler(CommandHandler("stop_trading", telegram_handler.handle_stop_trading))
            self.application.add_handler(CommandHandler("performance", telegram_handler.handle_performance))
            self.application.add_handler(CommandHandler("symbols", telegram_handler.handle_symbols))
            self.application.add_handler(CommandHandler("regimes", telegram_handler.handle_regimes))
            self.application.add_handler(CommandHandler("correlations", telegram_handler.handle_correlations))
            self.application.add_handler(CommandHandler("signals", telegram_handler.handle_signals))
            self.application.add_handler(CommandHandler("chart", telegram_handler.handle_chart))
            self.application.add_handler(CommandHandler("duration", telegram_handler.handle_duration_stats))
            self.application.add_handler(CommandHandler("intraday", telegram_handler.handle_intraday_stats))
            self.application.add_handler(CommandHandler("recent", telegram_handler.handle_recent_performance))
            self.application.add_handler(CommandHandler("backtest", telegram_handler.handle_backtest))
            self.application.add_handler(CommandHandler("optimize", telegram_handler.handle_optimize))
            await self.trading_engine.initialize_models()
            logger.info("Модели успешно инициализированы")
            await self.notification_manager.send_startup_notification(self.trading_engine)
        except Exception as e:
            logger.error(f"Ошибка инициализации бота: {e}")
            logger.error(traceback.format_exc())
            raise

    async def start(self):
        try:
            await self.setup()
            self.trading_engine.is_running = True
            self.trading_task = asyncio.create_task(
                enhanced_trading_loop(self.trading_engine, self.notification_manager)
            )
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            await asyncio.Event().wait()
        except Exception as e:
            logger.error(f"Ошибка запуска бота: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.stop()

    async def stop(self):
        try:
            if self.trading_engine:
                self.trading_engine.is_running = False
            if self.trading_task:
                self.trading_task.cancel()
                try:
                    await self.trading_task
                except asyncio.CancelledError:
                    pass
            if self.application and self.application.running:
                await self.application.updater.stop()
                await self.application.stop()
            logger.info("Бот остановлен")
        except Exception as e:
            logger.error(f"Ошибка при остановке бота: {e}")

async def enhanced_trading_loop(trading_engine: EnhancedTradingEngine,
                              notification_manager: 'EnhancedNotificationManager'):
    logger.info("Улучшенный торговый цикл запущен")
    last_performance_update = datetime.datetime.now()
    last_model_update = datetime.datetime.now()
    last_microstructure_update = datetime.datetime.now()
    microstructure_data = {}
    try:
        while trading_engine.is_running:
            try:
                current_time = datetime.datetime.now()
                if (current_time - trading_engine.monitor.daily_stats['last_reset']).days > 0:
                    await notification_manager.send_daily_report(trading_engine.monitor)
                    trading_engine.monitor.reset_daily_stats()
                if (current_time - last_model_update).total_seconds() > 3600 * 4:
                    logger.info("Переобучение моделей...")
                    await trading_engine.initialize_models()
                    last_model_update = current_time
                if (current_time - trading_engine.position_manager.correlation_analyzer.last_update).total_seconds() > 3600:
                    trading_engine.position_manager.correlation_analyzer.update_correlations(client)
                if (current_time - last_microstructure_update).total_seconds() > 300:
                    microstructure_data = {}
                    for symbol in SYMBOLS:
                        try:
                            order_book = get_order_book_data(client, symbol)
                            if order_book:
                                trading_engine.position_manager.microstructure_analyzer.update_order_book(symbol, order_book)
                            trades = get_recent_trades(client, symbol)
                            if not trades.empty:
                                trading_engine.position_manager.microstructure_analyzer.update_trades(symbol, trades)
                            metrics = trading_engine.position_manager.microstructure_analyzer.calculate_order_flow_metrics(symbol)
                            price_jump = trading_engine.position_manager.microstructure_analyzer.detect_price_jumps(symbol)
                            if price_jump:
                                metrics['price_jump'] = True
                            balance = get_usdt_balance(client)
                            if balance > 0:
                                test_order_size = balance * 0.02 / get_symbol_price(client, symbol)
                                market_impact = trading_engine.position_manager.microstructure_analyzer.calculate_market_impact(
                                    symbol, test_order_size
                                )
                                metrics['market_impact'] = market_impact
                            microstructure_data[symbol] = metrics
                        except Exception as e:
                            logger.error(f"Ошибка обновления микроструктуры для {symbol}: {e}")
                    last_microstructure_update = current_time
                current_prices = {}
                for symbol in trading_engine.position_manager.positions.keys():
                    price = get_symbol_price(client, symbol)
                    if price:
                        current_prices[symbol] = price
                await trading_engine.position_manager.update_trailing_stops(current_prices, microstructure_data)
                tasks = []
                for symbol in SYMBOLS:
                    if symbol in trading_engine.models:
                        symbol_microdata = microstructure_data.get(symbol, {})
                        task = asyncio.create_task(
                            trading_engine.process_symbol(symbol, symbol_microdata)
                        )
                        tasks.append(task)
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if (current_time - last_performance_update).total_seconds() > 1800:
                    await notification_manager.send_performance_update(trading_engine.monitor)
                    last_performance_update = current_time
                await asyncio.sleep(UPDATE_INTERVAL_SEC // 2)
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                logger.error(traceback.format_exc())
                await notification_manager.send_error_notification(f"Ошибка в торговом цикле: {e}")
                await asyncio.sleep(60)
    except asyncio.CancelledError:
        logger.info("Торговый цикл остановлен")
    finally:
        for symbol in list(trading_engine.position_manager.positions.keys()):
            price = get_symbol_price(client, symbol)
            if price:
                await trading_engine.position_manager.close_position(symbol, price, "Shutdown")

class EnhancedTelegramHandler:
    def __init__(self, trading_engine: EnhancedTradingEngine, monitor: EnhancedPerformanceMonitor):
        self.trading_engine = trading_engine
        self.monitor = monitor

    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_message = (
            "🤖 <b>Улучшенный криптовалютный торговый бот v2.1</b>\n\n"
            "🚀 <b>Новые функции:</b>\n"
            "• Ансамблевые ML модели (RF + XGBoost)\n"
            "• Адаптивное управление рисками\n"
            "• Детектор рыночных режимов\n"
            "• Анализ микроструктуры рынка\n"
            "• Оптимизация для краткосрочной торговли\n"
            "• Частичное закрытие позиций\n"
            "• Расширенная статистика\n"
            "• Улучшенный бэктестинг\n\n"
            "📋 <b>Команды:</b>\n"
            "/status - Текущий статус\n"
            "/stats - Общая статистика\n"
            "/performance - Детальная производительность\n"
            "/positions - Активные позиции\n"
            "/symbols - Статистика по символам\n"
            "/regimes - Статистика по режимам\n"
            "/correlations - Корреляции\n"
            "/signals - Текущие сигналы\n"
            "/chart SYMBOL - График символа\n"
            "/backtest SYMBOL - Бэктест символа\n"
            "/optimize SYMBOL - Оптимизация параметров\n"
            "/duration - Статистика по длительности\n"
            "/intraday - Внутридневная статистика\n"
            "/recent - Последние сделки\n"
            "/start_trading - Запуск торговли\n"
            "/stop_trading - Остановка торговли"
        )
        await update.message.reply_text(welcome_message, parse_mode='HTML')

    async def handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            balance = get_usdt_balance(self.trading_engine.client)
            active_positions = len(self.trading_engine.position_manager.positions)
            daily_profit = self.monitor.daily_stats['profit']
            if balance > 0:
                profit_percent = daily_profit / balance * 100
            else:
                profit_percent = 0.0
            positions_info = ""
            if self.trading_engine.position_manager.positions:
                positions_info = "\n\n<b>Активные позиции:</b>\n"
                for symbol, pos in self.trading_engine.position_manager.positions.items():
                    positions_info += f"• {symbol} {pos.side} ({pos.confidence_score:.2f})\n"
            status_msg = (
                "📊 <b>Текущий статус</b>\n\n"
                f"💰 Баланс: {balance:.2f} USDT\n"
                f"📈 Дневной профит: {daily_profit:.2f} USDT ({profit_percent:.2f}%)\n"
                f"📍 Активных позиций: {active_positions}/{self.trading_engine.position_manager.max_positions}\n"
                f"🤖 Бот: {'🟢 Работает' if self.trading_engine.is_running else '🔴 Остановлен'}\n"
                f"🧠 Моделей обучено: {len(self.trading_engine.models)}\n"
                f"🕐 Последнее обновление моделей: {self.trading_engine.last_model_update.strftime('%Y-%m-%d %H:%M')}"
                f"{positions_info}"
            )
            await update.message.reply_text(status_msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка получения статуса: {e}")
            await update.message.reply_text("❌ Ошибка получения статуса")

    async def handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            metrics = self.monitor.performance_metrics
            best_hours = self.monitor.get_best_trading_hours()[:3]
            best_hours_str = ""
            if best_hours:
                best_hours_str = "\n\n<b>Лучшие часы для торговли:</b>\n"
                for hour, stats in best_hours:
                    best_hours_str += f"• {hour:02d}:00 - {stats['avg_profit']:.2f} USDT/сделка\n"
            stats_msg = (
                "📊 <b>Общая статистика торговли</b>\n\n"
                f"📈 Всего сделок: {metrics.total_trades}\n"
                f"✅ Прибыльных: {metrics.winning_trades} ({metrics.win_rate:.1%})\n"
                f"❌ Убыточных: {metrics.losing_trades}\n"
                f"💰 Общая прибыль: {metrics.total_profit:.2f} USDT\n"
                f"📊 Profit Factor: {metrics.profit_factor:.2f}\n"
                f"💹 Средний выигрыш: {metrics.avg_win:.2f} USDT\n"
                f"📉 Средний проигрыш: {metrics.avg_loss:.2f} USDT\n"
                f"📉 Max Drawdown: {metrics.max_drawdown:.2f} USDT\n"
                f"📈 Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n"
                f"📊 Sortino Ratio: {metrics.sortino_ratio:.2f}\n"
                f"🔥 Max выигрышей подряд: {metrics.max_consecutive_wins}\n"
                f"❄️ Max проигрышей подряд: {metrics.max_consecutive_losses}"
                f"{best_hours_str}"
            )
            await update.message.reply_text(stats_msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            await update.message.reply_text("❌ Ошибка получения статистики")

    async def handle_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not self.trading_engine.position_manager.positions:
                await update.message.reply_text("📊 Нет активных позиций")
                return
            positions_msg = "💼 <b>Активные позиции</b>\n\n"
            for symbol, pos in self.trading_engine.position_manager.positions.items():
                current_price = get_symbol_price(self.trading_engine.client, symbol)
                if not current_price:
                    continue
                if pos.side == "LONG":
                    current_profit = pos.quantity * (current_price - pos.entry_price)
                    profit_percent = (current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    current_profit = pos.quantity * (pos.entry_price - current_price)
                    profit_percent = (pos.entry_price - current_price) / pos.entry_price * 100
                profit_emoji = "🟢" if current_profit > 0 else "🔴"
                positions_msg += (
                    f"{profit_emoji} <b>{symbol}</b> {pos.side}\n"
                    f"  📍 Вход: {pos.entry_price:.8f}\n"
                    f"  💵 Текущая: {current_price:.8f}\n"
                    f"  📊 Количество: {pos.quantity:.8f}\n"
                    f"  💰 П/У: {current_profit:.2f} USDT ({profit_percent:.2f}%)\n"
                    f"  🛡 SL: {pos.stop_loss:.8f} | TP: {pos.take_profit:.8f}\n"
                    f"  🌐 Режим: {pos.market_regime}\n"
                    f"  🎯 Уверенность: {pos.confidence_score:.2f}\n"
                    f"  📈 Max прибыль: {pos.max_profit:.2%}\n"
                    f"  ⏱ Время: {(datetime.datetime.now() - pos.entry_time).total_seconds() / 60:.0f} мин\n\n"
                )
            await update.message.reply_text(positions_msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка получения позиций: {e}")
            await update.message.reply_text("❌ Ошибка получения позиций")

    async def handle_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not self.monitor.trade_history:
                await update.message.reply_text("📊 История торговли пуста")
                return
            recent_trades = self.monitor.trade_history[-10:]
            history_msg = "📜 <b>История последних сделок</b>\n\n"
            for i, trade in enumerate(reversed(recent_trades), 1):
                profit_emoji = "🟢" if trade['profit'] > 0 else "🔴"
                history_msg += (
                    f"{i}. {profit_emoji} <b>{trade['symbol']}</b> {trade.get('side', 'N/A')}\n"
                    f"   💰 Прибыль: {trade['profit']:.2f} USDT ({trade.get('profit_percent', 0):.2f}%)\n"
                    f"   🕐 Время: {trade['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
                    f"   🌐 Режим: {trade.get('market_regime', 'N/A')}\n"
                    f"   📝 Причина закрытия: {trade.get('reason', 'N/A')}\n\n"
                )
            await update.message.reply_text(history_msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка получения истории: {e}")
            await update.message.reply_text("❌ Ошибка получения истории")

    async def handle_start_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if self.trading_engine.is_running:
                await update.message.reply_text("🟢 Бот уже запущен")
                return
            self.trading_engine.is_running = True
            await update.message.reply_text("🚀 Торговля запущена")
            logger.info("Торговля запущена через Telegram")
        except Exception as e:
            logger.error(f"Ошибка запуска торговли: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_stop_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not self.trading_engine.is_running:
                await update.message.reply_text("🔴 Бот уже остановлен")
                return
            self.trading_engine.is_running = False
            if self.trading_engine.position_manager.positions:
                await update.message.reply_text("⏳ Закрытие всех позиций...")
                for symbol in list(self.trading_engine.position_manager.positions.keys()):
                    price = get_symbol_price(self.trading_engine.client, symbol)
                    if price:
                        await self.trading_engine.position_manager.close_position(
                            symbol, price, "Manual stop"
                        )
            await update.message.reply_text("🛑 Торговля остановлена")
            logger.info("Торговля остановлена через Telegram")
        except Exception as e:
            logger.error(f"Ошибка остановки торговли: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            metrics = self.monitor.performance_metrics
            if self.monitor.trade_history:
                plt.figure(figsize=(10, 6))
                profits = [t['profit'] for t in self.monitor.trade_history]
                cumulative_profits = np.cumsum(profits)
                plt.plot(cumulative_profits, label='Кумулятивная прибыль')
                plt.title('Кривая эквити')
                plt.xlabel('Количество сделок')
                plt.ylabel('Прибыль (USDT)')
                plt.grid(True)
                plt.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()
                caption = (
                    f"📊 <b>Детальная статистика</b>\n\n"
                    f"📈 Всего сделок: {metrics.total_trades}\n"
                    f"✅ Прибыльных: {metrics.winning_trades}\n"
                    f"❌ Убыточных: {metrics.losing_trades}\n"
                    f"🎯 Win Rate: {metrics.win_rate:.2%}\n"
                    f"💰 Общая прибыль: {metrics.total_profit:.2f} USDT\n"
                    f"📊 Profit Factor: {metrics.profit_factor:.2f}\n"
                    f"📈 Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n"
                    f"📉 Sortino Ratio: {metrics.sortino_ratio:.2f}\n"
                    f"💹 Calmar Ratio: {metrics.calmar_ratio:.2f}\n"
                    f"📉 Max Drawdown: {metrics.max_drawdown:.2f} USDT\n"
                    f"🔥 Max подряд выигрышей: {metrics.max_consecutive_wins}\n"
                    f"❄️ Max подряд проигрышей: {metrics.max_consecutive_losses}"
                )
                await update.message.reply_photo(photo=buf, caption=caption, parse_mode='HTML')
            else:
                await update.message.reply_text("📊 Нет данных для отображения")
        except Exception as e:
            logger.error(f"Ошибка отображения производительности: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            best_symbols = self.monitor.get_best_symbols(10)
            if not best_symbols:
                await update.message.reply_text("📊 Нет данных по символам")
                return
            msg = "💎 <b>Статистика по символам</b>\n\n"
            for i, (symbol, stats) in enumerate(best_symbols, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔸"
                msg += (
                    f"{emoji} <b>{symbol}</b>\n"
                    f"  📊 Сделок: {stats['trades']}\n"
                    f"  💰 Прибыль: {stats['profit']:.2f} USDT\n"
                    f"  🎯 Win Rate: {stats['win_rate']:.2%}\n"
                    f"  📈 Ср. прибыль: {stats['avg_profit']:.2f} USDT\n\n"
                )
            await update.message.reply_text(msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка отображения статистики символов: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_regimes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            regime_perf = self.monitor.get_regime_performance()
            if not regime_perf:
                await update.message.reply_text("📊 Нет данных по режимам рынка")
                return
            msg = "🌐 <b>Производительность по режимам рынка</b>\n\n"
            regime_emojis = {
                'trending_up': '📈',
                'trending_down': '📉',
                'ranging': '➡️',
                'high_volatility': '🌊',
                'unknown': '❓'
            }
            for regime, stats in regime_perf.items():
                emoji = regime_emojis.get(regime, '🔸')
                msg += (
                    f"{emoji} <b>{regime.replace('_', ' ').title()}</b>\n"
                    f"  📊 Сделок: {stats['trades']}\n"
                    f"  💰 Прибыль: {stats['profit']:.2f} USDT\n"
                    f"  🎯 Win Rate: {stats['win_rate']:.2%}\n"
                    f"  📈 Ср. прибыль: {stats['avg_profit']:.2f} USDT\n\n"
                )
            await update.message.reply_text(msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка отображения статистики режимов: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_correlations(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            correlations = self.trading_engine.position_manager.correlation_analyzer.correlation_cache
            if correlations.empty:
                correlations = self.trading_engine.position_manager.correlation_analyzer.update_correlations(
                    self.trading_engine.client
                )
            if correlations.empty:
                await update.message.reply_text("❌ Не удалось рассчитать корреляции")
                return
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Корреляционная матрица символов')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            high_corr = []
            for i in range(len(correlations)):
                for j in range(i+1, len(correlations)):
                    corr_value = correlations.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_corr.append((
                            correlations.index[i],
                            correlations.columns[j],
                            corr_value
                        ))
            caption = "🔗 <b>Корреляции между символами</b>\n\n"
            if high_corr:
                caption += "⚠️ <b>Высокие корреляции:</b>\n"
                for sym1, sym2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
                    caption += f"• {sym1} ↔️ {sym2}: {corr:.3f}\n"
            else:
                caption += "✅ Нет высоких корреляций (>0.7)"
            await update.message.reply_photo(photo=buf, caption=caption, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка отображения корреляций: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            signals_msg = "📡 <b>Текущие торговые сигналы</b>\n\n"
            for symbol in SYMBOLS:
                if symbol not in self.trading_engine.models:
                    continue
                df = fetch_binance_klines(self.trading_engine.client, symbol, TIMEFRAME, 200)
                if df.empty:
                    continue
                df = calculate_enhanced_indicators(df)
                if df.empty:
                    continue
                market_conditions = self.trading_engine.market_detector.detect_regime(df)
                model_data = self.trading_engine.models[symbol]
                if len(model_data) == 6:
                    model, scaler, features, threshold, _, all_features = model_data
                else:
                    model, scaler, features, threshold, _ = model_data
                    all_features = None
                label, prob = predict_enhanced_direction(
                    df, model, scaler, features, self.trading_engine.commission, threshold, all_features
                )
                confidence_score = abs(prob - 0.5) * 2
                signal, strength, reason, final_confidence = self.trading_engine.signal_analyzer.analyze_signal(
                    df, prob, market_conditions, confidence_score
                )
                signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
                signals_msg += (
                    f"{signal_emoji} <b>{symbol}</b>\n"
                    f"  📊 Сигнал: {signal}\n"
                    f"  🎯 Уверенность: {final_confidence:.2f}\n"
                    f"  🌐 Режим: {market_conditions.regime}\n"
                    f"  📝 Причина: {reason}\n\n"
                )
            if signals_msg == "📡 <b>Текущие торговые сигналы</b>\n\n":
                signals_msg = "📊 Нет активных сигналов"
            await update.message.reply_text(signals_msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка получения сигналов: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            args = context.args
            if not args:
                await update.message.reply_text("❌ Укажите символ, например: /chart BTCUSDT")
                return
            symbol = args[0].upper()
            if symbol not in SYMBOLS:
                await update.message.reply_text(f"❌ Символ {symbol} не поддерживается")
                return
            df = fetch_binance_klines(self.trading_engine.client, symbol, TIMEFRAME, 200)
            if df.empty:
                await update.message.reply_text(f"❌ Не удалось загрузить данные для {symbol}")
                return
            plt.figure(figsize=(10, 6))
            plt.plot(df['close'], label='Цена закрытия')
            plt.title(f'График цен {symbol}')
            plt.xlabel('Время')
            plt.ylabel('Цена (USDT)')
            plt.grid(True)
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            await update.message.reply_photo(
                photo=buf,
                caption=f"📈 График цен для {symbol}",
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Ошибка построения графика: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_duration_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            duration_stats = self.monitor.get_duration_performance()
            if not duration_stats:
                await update.message.reply_text("📊 Нет данных по длительности сделок")
                return
            msg = "⏱ <b>Статистика по длительности сделок</b>\n\n"
            duration_labels = {
                'very_short': 'Очень короткие (<15 мин)',
                'short': 'Короткие (15-60 мин)',
                'medium': 'Средние (1-4 ч)',
                'long': 'Долгие (>4 ч)'
            }
            for category, stats in duration_stats.items():
                msg += (
                    f"🔸 <b>{duration_labels[category]}</b>\n"
                    f"  📊 Сделок: {stats['trades']}\n"
                    f"  💰 Прибыль: {stats['profit']:.2f} USDT\n"
                    f"  🎯 Win Rate: {stats['win_rate']:.2%}\n"
                    f"  📈 Ср. прибыль: {stats['avg_profit']:.2f} USDT\n\n"
                )
            await update.message.reply_text(msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка статистики длительности: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_intraday_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            intraday_stats = self.monitor.get_intraday_performance()
            if not intraday_stats:
                await update.message.reply_text("📊 Нет внутридневной статистики")
                return
            msg = "🌞 <b>Внутридневная статистика</b>\n\n"
            for hour, stats in intraday_stats:
                msg += (
                    f"🕒 <b>{hour:02d}:00</b>\n"
                    f"  📊 Сделок: {stats['trades']}\n"
                    f"  💰 Прибыль: {stats['profit']:.2f} USDT\n"
                    f"  📈 Ср. прибыль: {stats['avg_profit']:.2f} USDT\n\n"
                )
            await update.message.reply_text(msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка внутридневной статистики: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_recent_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            recent_perf = self.monitor.get_recent_performance()
            msg = "⏳ <b>Производительность последних сделок</b>\n\n"
            msg += (
                f"📊 Сделок: {recent_perf['trades']}\n"
                f"🎯 Win Rate: {recent_perf['win_rate']:.2%}\n"
                f"💰 Общая прибыль: {recent_perf['total_profit']:.2f} USDT\n"
                f"📈 Ср. прибыль: {recent_perf['avg_profit']:.2f} USDT\n"
            )
            await update.message.reply_text(msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка получения последних сделок: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            args = context.args
            if not args:
                await update.message.reply_text("❌ Укажите символ, например: /backtest BTCUSDT")
                return
            symbol = args[0].upper()
            if symbol not in SYMBOLS:
                await update.message.reply_text(f"❌ Символ {symbol} не поддерживается")
                return
            if symbol not in self.trading_engine.models:
                await update.message.reply_text(f"❌ Модель для {symbol} не найдена")
                return
            model_data = self.trading_engine.models[symbol]
            if len(model_data) == 6:
                model, scaler, features, threshold, _, all_features = model_data
            else:
                model, scaler, features, threshold, _ = model_data
                all_features = None
            df = fetch_binance_klines(self.trading_engine.client, symbol, TIMEFRAME, 1000)
            if df.empty:
                await update.message.reply_text(f"❌ Не удалось загрузить данные для {symbol}")
                return
            backtester = Backtester()
            results = backtester.backtest_strategy(
                df, model, scaler, features,
                self.trading_engine.signal_analyzer,
                self.trading_engine.market_detector,
                self.trading_engine.threshold_manager,
                threshold,
                all_features
            )
            msg = (
                f"📊 <b>Результаты бэктеста для {symbol}</b>\n\n"
                f"📈 Сделок: {results['total_trades']}\n"
                f"🎯 Win Rate: {results['win_rate']:.2%}\n"
                f"💰 Profit Factor: {results['profit_factor']:.2f}\n"
                f"📉 Max Drawdown: {results['max_drawdown']:.2%}\n"
                f"📈 Total Return: {results['total_return']:.2%}\n"
                f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                f"🔥 Max подряд выигрышей: {results['max_consecutive_wins']}\n"
                f"❄️ Max подряд проигрышей: {results['max_consecutive_losses']}\n"
            )
            await update.message.reply_text(msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Ошибка бэктеста: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

    async def handle_optimize(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            args = context.args
            if not args:
                await update.message.reply_text("❌ Укажите символ, например: /optimize BTCUSDT")
                return
            symbol = args[0].upper()
            if symbol not in SYMBOLS:
                await update.message.reply_text(f"❌ Символ {symbol} не поддерживается")
                return
            success = await self.trading_engine.optimize_model_parameters(symbol)
            if success:
                await update.message.reply_text(f"✅ Параметры для {symbol} оптимизированы")
            else:
                await update.message.reply_text(f"⚠️ Не удалось оптимизировать параметры для {symbol}")
        except Exception as e:
            logger.error(f"Ошибка оптимизации: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")

class EnhancedNotificationManager:
    def __init__(self, bot, chat_id):
        self.bot = bot
        self.chat_id = chat_id
        self.last_notification = {}
        self.notification_cooldown = 300  # 5 минут

    async def send_notification(self, message: str, parse_mode: str = 'HTML', notification_type: str = 'general'):
        """Отправка уведомления с защитой от спама"""
        try:
            current_time = time.time()
            if notification_type in self.last_notification:
                if current_time - self.last_notification[notification_type] < self.notification_cooldown:
                    return
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            self.last_notification[notification_type] = current_time
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления: {e}")

    async def send_trade_notification(self, position: Position, action: str):
        """Уведомление о торговых операциях"""
        try:
            if action == "OPEN":
                emoji = "🟢" if position.side == "LONG" else "🔴"
                message = (
                    f"{emoji} <b>Открыта позиция</b>\n\n"
                    f"💎 Символ: {position.symbol}\n"
                    f"📍 Направление: {position.side}\n"
                    f"💵 Цена входа: {position.entry_price:.8f}\n"
                    f"📊 Количество: {position.quantity:.8f}\n"
                    f"🎯 Take Profit: {position.take_profit:.8f}\n"
                    f"🛡 Stop Loss: {position.stop_loss:.8f}\n"
                    f"🌐 Режим рынка: {position.market_regime}\n"
                    f"🎯 Уверенность: {position.confidence_score:.2f}"
                )
            elif action == "CLOSE":
                message = f"🔒 Позиция {position.symbol} закрыта"
            await self.send_notification(message, notification_type='trade')
        except Exception as e:
            logger.error(f"Ошибка отправки торгового уведомления: {e}")

    async def send_performance_update(self, monitor: EnhancedPerformanceMonitor):
        """Периодическое обновление производительности"""
        try:
            metrics = monitor.performance_metrics
            message = (
                "📊 <b>Обновление производительности</b>\n\n"
                f"📈 Сделок за сегодня: {monitor.daily_stats['trades']}\n"
                f"💰 Дневная прибыль: {monitor.daily_stats['profit']:.2f} USDT\n"
                f"🎯 Win Rate: {metrics.win_rate:.1%}\n"
                f"📊 Profit Factor: {metrics.profit_factor:.2f}\n"
                f"💵 Общая прибыль: {metrics.total_profit:.2f} USDT"
            )
            await self.send_notification(message, notification_type='performance')
        except Exception as e:
            logger.error(f"Ошибка отправки обновления производительности: {e}")

    async def send_daily_report(self, monitor: EnhancedPerformanceMonitor):
        """Ежедневный отчет"""
        try:
            daily = monitor.daily_stats
            metrics = monitor.performance_metrics
            best_symbols = monitor.get_best_symbols(3)
            symbols_text = ""
            if best_symbols:
                symbols_text = "\n\n<b>Лучшие символы дня:</b>\n"
                for symbol, stats in best_symbols:
                    symbols_text += f"• {symbol}: {stats['profit']:.2f} USDT\n"
            message = (
                "📅 <b>Ежедневный отчет</b>\n\n"
                f"📆 Дата: {datetime.datetime.now().strftime('%Y-%m-%d')}\n"
                f"📊 Сделок: {daily['trades']}\n"
                f"💰 Прибыль: {daily['profit']:.2f} USDT\n"
                f"📈 ROI: {(daily['profit'] / daily['start_balance'] * 100) if daily['start_balance'] > 0 else 0:.2f}%\n"
                f"🎯 Win Rate: {metrics.win_rate:.1%}\n"
                f"💵 Общая прибыль: {metrics.total_profit:.2f} USDT"
                f"{symbols_text}"
            )
            await self.send_notification(message, notification_type='daily_report')
        except Exception as e:
            logger.error(f"Ошибка отправки дневного отчета: {e}")

    async def send_error_notification(self, error_message: str):
        """Уведомление об ошибках"""
        try:
            message = f"⚠️ <b>Ошибка в работе бота</b>\n\n{error_message}"
            await self.send_notification(message, notification_type='error')
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления об ошибке: {e}")

    async def send_startup_notification(self, trading_engine: EnhancedTradingEngine):
        """Уведомление о запуске бота"""
        try:
            balance = get_usdt_balance(trading_engine.client)
            message = (
                "🚀 <b>Бот успешно запущен</b>\n\n"
                f"💰 Начальный баланс: {balance:.2f} USDT\n"
                f"🧠 Моделей обучено: {len(trading_engine.models)}\n"
                f"💎 Символов: {', '.join(trading_engine.models.keys())}\n"
                f"🔧 Версия: 2.0 Enhanced\n"
                f"🕐 Время: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await self.send_notification(message, notification_type='startup')
        except Exception as e:
            logger.error(f"Ошибка отправки уведомления о запуске: {e}")

async def main():
    """Основная функция запуска улучшенного бота"""
    try:
        position_manager = EnhancedPositionManager(client)
        signal_analyzer = EnhancedSignalAnalyzer(COMMISSION, SPREAD)
        monitor = EnhancedPerformanceMonitor()
        trading_engine = EnhancedTradingEngine(client, position_manager, signal_analyzer, monitor)
        logger.info("Инициализация и валидация моделей...")
        await trading_engine.initialize_models()
        logger.info("Запуск торгового цикла...")
        trading_engine.is_running = True
        while trading_engine.is_running:
            try:
                if (datetime.datetime.now() - trading_engine.last_model_update).total_seconds() > 3600 * 6:
                    logger.info("Обновление моделей...")
                    await trading_engine.initialize_models()
                if (datetime.datetime.now() - position_manager.correlation_analyzer.last_update).total_seconds() > 3600:
                    position_manager.correlation_analyzer.update_correlations(client)
                current_prices = {}
                for symbol in position_manager.positions.keys():
                    price = get_symbol_price(client, symbol)
                    if price:
                        current_prices[symbol] = price
                await position_manager.update_trailing_stops(current_prices)
                for symbol in SYMBOLS:
                    if symbol in trading_engine.models:
                        await trading_engine.process_symbol(symbol)
                        await asyncio.sleep(1)
                if datetime.datetime.now().minute % 30 == 0:
                    logger.info(f"=== Отчет о производительности ===")
                    logger.info(f"Открытых позиций: {len(position_manager.positions)}")
                    logger.info(f"Общая прибыль: {monitor.performance_metrics.total_profit:.2f} USDT")
                    logger.info(f"Win Rate: {monitor.performance_metrics.win_rate:.2%}")
                    logger.info(f"Profit Factor: {monitor.performance_metrics.profit_factor:.2f}")
                    best_symbols = monitor.get_best_symbols(3)
                    if best_symbols:
                        logger.info("Лучшие символы:")
                        for symbol, stats in best_symbols:
                            logger.info(f"  {symbol}: {stats['profit']:.2f} USDT, WR: {stats['win_rate']:.2%}")
                    regime_perf = monitor.get_regime_performance()
                    if regime_perf:
                        logger.info("Производительность по режимам:")
                        for regime, stats in regime_perf.items():
                            logger.info(f"  {regime}: {stats['profit']:.2f} USDT, WR: {stats['win_rate']:.2%}")
                await asyncio.sleep(UPDATE_INTERVAL_SEC)
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Закрытие всех открытых позиций...")
        for symbol in list(position_manager.positions.keys()):
            price = get_symbol_price(client, symbol)
            if price:
                await position_manager.close_position(symbol, price, "Shutdown")
        logger.info("Бот остановлен")

if __name__ == "__main__":
    try:
        if not config.get('telegram', {}).get('token'):
            logger.error("Telegram токен не настроен")
            sys.exit(1)
        if not config.get('binance', {}).get('api_key'):
            logger.error("Binance API ключи не настроены")
            sys.exit(1)
        if len(sys.argv) > 1 and sys.argv[1] == "--no-telegram":
            logger.info("Запуск в режиме без Telegram...")
            asyncio.run(main())
        else:
            logger.info("Запуск с Telegram интеграцией...")
            bot = EnhancedTelegramBot()
            asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        logger.error(traceback.format_exc())