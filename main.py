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

# Импорт из наших модулей
from calculations import (
    Position, TradeStats, MarketConditions, TechnicalIndicators,
    MarketRegimeDetector, StopLossOptimizer, CorrelationAnalyzer,
    calculate_enhanced_indicators, prepare_enhanced_features
)
from ml_models import (
    EnsembleModel, AdaptiveThresholdManager, EnhancedSignalAnalyzer,
    train_enhanced_model, predict_enhanced_direction, Backtester
)
from binance_utils import (
    fetch_binance_klines, get_usdt_balance, get_symbol_price,
    get_lot_size, place_market_order, place_stop_loss_order,
    place_take_profit_order, cancel_order, get_price_precision
)

# Фикс для Windows
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())
nest_asyncio.apply()

# Установка UTF-8 для консоли Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- Конфигурация логгера ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot_enhanced.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CryptoBotEnhanced")

# --- Загрузка конфигурации ---
def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        raise

config = load_config()

# --- Инициализация клиентов ---
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

# --- Константы ---
SYMBOLS = config['trading']['symbols']
TIMEFRAME = config['trading']['timeframe']
RISK_USD = config['trading']['risk_usd']
COMMISSION = config['trading']['commission']
SPREAD = config['trading']['spread']
HISTORICAL_DAYS = config['trading']['historical_days']
UPDATE_INTERVAL_SEC = int(config['trading']['update_interval_sec'])

# --- Улучшенный мониторинг производительности ---
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
        
    def update_trade_stats(self, trade_info: dict):
        """Обновление статистики торговли"""
        try:
            self.trade_history.append(trade_info)
            
            # Обновление дневной статистики
            self.daily_stats['trades'] += 1
            self.daily_stats['profit'] += trade_info.get('profit', 0)
            
            # Обновление статистики по символам
            symbol = trade_info['symbol']
            self.symbol_stats[symbol]['trades'] += 1
            self.symbol_stats[symbol]['profit'] += trade_info.get('profit', 0)
            if trade_info.get('profit', 0) > 0:
                self.symbol_stats[symbol]['wins'] += 1
                
            # Обновление статистики по часам
            hour = trade_info['timestamp'].hour
            self.hourly_stats[hour]['trades'] += 1
            self.hourly_stats[hour]['profit'] += trade_info.get('profit', 0)
            
            # Обновление статистики по режимам рынка
            regime = trade_info.get('market_regime', 'unknown')
            self.regime_stats[regime]['trades'] += 1
            self.regime_stats[regime]['profit'] += trade_info.get('profit', 0)
            if trade_info.get('profit', 0) > 0:
                self.regime_stats[regime]['wins'] += 1
            
            # Обновление общих метрик
            self.calculate_performance_metrics()
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
            logger.error(traceback.format_exc())
    
    def calculate_performance_metrics(self):
        """Расчет расширенных метрик эффективности"""
        try:
            if not self.trade_history:
                return
                
            profits = [t.get('profit', 0) for t in self.trade_history]
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p < 0]
            
            # Базовые метрики
            self.performance_metrics.total_trades = len(profits)
            self.performance_metrics.winning_trades = len(winning_trades)
            self.performance_metrics.losing_trades = len(losing_trades)
            self.performance_metrics.total_profit = sum(profits)
            
            # Win rate
            self.performance_metrics.win_rate = len(winning_trades) / len(profits) if profits else 0
            
            # Средние значения
            self.performance_metrics.avg_win = np.mean(winning_trades) if winning_trades else 0
            self.performance_metrics.avg_loss = np.mean(losing_trades) if losing_trades else 0
            
            # Profit factor
            total_wins = sum(winning_trades) if winning_trades else 0.0001
            total_losses = abs(sum(losing_trades)) if losing_trades else 0.0001
            self.performance_metrics.profit_factor = total_wins / total_losses
            
            # Максимальная просадка
            self.performance_metrics.max_drawdown = self._calculate_max_drawdown(profits)
            
            # Sharpe ratio
            self.performance_metrics.sharpe_ratio = self._calculate_sharpe_ratio(profits)
            
            # Sortino ratio
            self.performance_metrics.sortino_ratio = self._calculate_sortino_ratio(profits)
            
            # Calmar ratio
            if self.performance_metrics.max_drawdown > 0:
                annual_return = self.performance_metrics.total_profit * 365 / len(profits) if len(profits) > 0 else 0
                self.performance_metrics.calmar_ratio = annual_return / self.performance_metrics.max_drawdown
            
            # Последовательные выигрыши/проигрыши
            self._calculate_consecutive_stats(profits)
            
        except Exception as e:
            logger.error(f"Ошибка расчета метрик: {e}")
            logger.error(traceback.format_exc())
    
    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Расчет максимальной просадки"""
        if not profits:
            return 0.0
            
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        
        return np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    def _calculate_sharpe_ratio(self, profits: List[float]) -> float:
        """Расчет коэффициента Шарпа"""
        if not profits or len(profits) < 2:
            return 0.0
            
        returns = np.array(profits)
        if np.std(returns) == 0:
            return 0.0
            
        # Аннуализированный Sharpe ratio
        return np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 4)  # для 15-минутных интервалов
    
    def _calculate_sortino_ratio(self, profits: List[float]) -> float:
        """Расчет коэффициента Сортино"""
        if not profits or len(profits) < 2:
            return 0.0
            
        returns = np.array(profits)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
            
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
            
        return np.mean(returns) / downside_std * np.sqrt(365 * 24 * 4)
    
    def _calculate_consecutive_stats(self, profits: List[float]):
        """Расчет последовательных выигрышей/проигрышей"""
        if not profits:
            return
            
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for profit in profits:
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
                
        self.performance_metrics.max_consecutive_wins = max_wins
        self.performance_metrics.max_consecutive_losses = max_losses
    
    def get_best_symbols(self, top_n: int = 5) -> List[Tuple[str, dict]]:
        """Получение лучших символов по производительности"""
        symbol_performance = []
        
        for symbol, stats in self.symbol_stats.items():
            if stats['trades'] > 0:
                win_rate = stats['wins'] / stats['trades']
                avg_profit = stats['profit'] / stats['trades']
                score = win_rate * avg_profit  # Простая оценка
                
                symbol_performance.append((
                    symbol,
                    {
                        'trades': stats['trades'],
                        'profit': stats['profit'],
                        'win_rate': win_rate,
                        'avg_profit': avg_profit,
                        'score': score
                    }
                ))
                
        return sorted(symbol_performance, key=lambda x: x[1]['score'], reverse=True)[:top_n]
    
    def get_best_trading_hours(self) -> List[Tuple[int, dict]]:
        """Получение лучших часов для торговли"""
        hour_performance = []
        
        for hour, stats in self.hourly_stats.items():
            if stats['trades'] > 0:
                avg_profit = stats['profit'] / stats['trades']
                hour_performance.append((
                    hour,
                    {
                        'trades': stats['trades'],
                        'total_profit': stats['profit'],
                        'avg_profit': avg_profit
                    }
                ))
                
        return sorted(hour_performance, key=lambda x: x[1]['avg_profit'], reverse=True)
    
    def get_regime_performance(self) -> Dict[str, dict]:
        """Получение производительности по режимам рынка"""
        regime_performance = {}
        
        for regime, stats in self.regime_stats.items():
            if stats['trades'] > 0:
                regime_performance[regime] = {
                    'trades': stats['trades'],
                    'profit': stats['profit'],
                    'win_rate': stats['wins'] / stats['trades'],
                    'avg_profit': stats['profit'] / stats['trades']
                }
                
        return regime_performance
    
    def reset_daily_stats(self):
        """Сброс дневной статистики"""
        self.daily_stats = {
            'trades': 0,
            'profit': 0.0,
            'start_balance': get_usdt_balance(client),
            'max_balance': 0.0,
            'min_balance': float('inf'),
            'last_reset': datetime.datetime.now()
        }

# --- Улучшенный менеджер позиций ---
class EnhancedPositionManager:
    def __init__(self, client, risk_percent=0.01, max_positions=5, max_correlation=0.7):
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
        
    def can_open_position(self, symbol: str) -> bool:
        """Проверяет возможность открытия новой позиции"""
        if len(self.positions) >= self.max_positions:
            logger.info(f"Достигнуто максимальное количество позиций: {self.max_positions}")
            return False
            
        if symbol in self.positions:
            logger.info(f"Позиция для {symbol} уже открыта")
            return False
            
        # Проверка корреляций
        if not self.correlation_analyzer.should_trade_symbol(symbol, self.positions, self.max_correlation):
            logger.info(f"Высокая корреляция {symbol} с открытыми позициями")
            return False
            
        return True
    
    def calculate_dynamic_position_size(self, symbol: str, price: float, atr: float, 
                                      market_conditions: MarketConditions, 
                                      confidence_score: float) -> Tuple[float, float]:
        """Динамический расчет размера позиции"""
        try:
            balance = get_usdt_balance(self.client)
            if balance is None or balance <= 0:
                logger.warning(f"Недостаточный баланс: {balance}")
                return 0.0, 0.0
            
            # Базовый риск
            base_risk = self.risk_percent
            
            # Корректировка риска по режиму рынка
            risk_multipliers = {
                "high_volatility": 0.5,
                "trending_up": 1.2,
                "trending_down": 1.2,
                "ranging": 0.7,
                "unknown": 0.8
            }
            market_multiplier = risk_multipliers.get(market_conditions.regime, 1.0)
            
            # Корректировка по волатильности
            volatility_multiplier = 1.0
            if market_conditions.volatility_percentile > 80:
                volatility_multiplier = 0.7
            elif market_conditions.volatility_percentile < 20:
                volatility_multiplier = 1.3
                
            # Корректировка по уверенности модели
            confidence_multiplier = 0.5 + confidence_score
            
            # Корректировка по текущей производительности
            performance_multiplier = self._get_performance_multiplier()
            
            # Итоговый риск
            adjusted_risk = base_risk * market_multiplier * volatility_multiplier * \
                          confidence_multiplier * performance_multiplier
            adjusted_risk = np.clip(adjusted_risk, 0.005, 0.03)  # 0.5% - 3%
            
            # Расчет размера позиции
            risk_amount = balance * adjusted_risk
            stop_distance = atr * self.stop_loss_optimizer.atr_multipliers.get(market_conditions.regime, 2.0)
            
            position_size_usd = risk_amount
            position_size = position_size_usd / price
            
            # Проверка ограничений
            min_qty, qty_precision = get_lot_size(self.client, symbol)
            position_size = max(round(position_size, qty_precision), min_qty)
            
            # Максимальный размер позиции - 10% от баланса
            max_position = balance * 0.1 / price
            position_size = min(position_size, max_position)
            
            logger.info(f"Рассчитан размер позиции для {symbol}: {position_size} "
                       f"(${position_size*price:.2f}), риск: {adjusted_risk*100:.2f}%")
            
            return position_size, stop_distance
            
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            logger.error(traceback.format_exc())
            return 0.0, 0.0
    
    def _get_performance_multiplier(self) -> float:
        """Получение множителя на основе производительности"""
        if len(self.position_history) < 10:
            return 1.0
            
        recent_trades = self.position_history[-20:]
        win_rate = sum(1 for t in recent_trades if t['profit'] > 0) / len(recent_trades)
        
        if win_rate > 0.65:
            return 1.2
        elif win_rate < 0.35:
            return 0.8
        else:
            return 1.0
    
    async def open_position(self, symbol: str, side: str, price: float, quantity: float,
                          stop_loss: float, take_profit: float, market_conditions: MarketConditions,
                          confidence_score: float) -> Optional[Position]:
        """Открытие новой позиции с улучшенными параметрами"""
        try:
            if not self.can_open_position(symbol):
                return None
                
            # Проверка минимального размера позиции
            min_qty, _ = get_lot_size(self.client, symbol)
            if quantity < min_qty:
                logger.warning(f"Размер позиции {quantity} меньше минимального {min_qty} для {symbol}")
                return None
                
            # Размещение рыночного ордера
            order_side = "BUY" if side == "LONG" else "SELL"
            order = place_market_order(self.client, symbol, order_side, quantity)
            
            if not order:
                logger.error(f"Не удалось разместить ордер для {symbol} {side}")
                return None
                
            # Получение фактической цены исполнения
            fills = order.get('fills', [])
            if fills:
                actual_price = float(fills[0]['price'])
            else:
                actual_price = price
                
            # Создание объекта позиции
            position = Position(
                symbol=symbol,
                entry_price=actual_price,
                quantity=quantity,
                side=side,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=stop_loss,
                trailing_take_profit=take_profit,
                entry_time=datetime.datetime.now(),
                order_id=order['orderId'],
                market_regime=market_conditions.regime,
                confidence_score=confidence_score
            )
            
            # Размещение стоп-лосс и тейк-профит ордеров
            await self.place_sl_tp_orders(position)
            
            self.positions[symbol] = position
            logger.info(f"Открыта позиция: {symbol} {side} {quantity} по цене {actual_price}, "
                       f"режим: {market_conditions.regime}, уверенность: {confidence_score:.2f}")
            
            return position
            
        except Exception as e:
            logger.error(f"Ошибка открытия позиции: {e}")
            logger.error(traceback.format_exc())
            return None
    
    async def place_sl_tp_orders(self, position: Position):
        """Размещение стоп-лосс и тейк-профит ордеров"""
        try:
            close_side = "SELL" if position.side == "LONG" else "BUY"
            
            # Размещение стоп-лосса
            sl_order = place_stop_loss_order(
                self.client, position.symbol, close_side, 
                position.quantity, position.stop_loss
            )
            if sl_order:
                self.stop_loss_orders[position.symbol] = sl_order['orderId']
            
            # Размещение тейк-профита
            tp_order = place_take_profit_order(
                self.client, position.symbol, close_side,
                position.quantity, position.take_profit
            )
            if tp_order:
                self.take_profit_orders[position.symbol] = tp_order['orderId']
                
        except Exception as e:
            logger.error(f"Ошибка размещения SL/TP ордеров: {e}")
    
    async def update_trailing_stops(self, current_prices: Dict[str, float]):
        """Обновление трейлинг-стопов и трейлинг-тейк-профитов"""
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Обновление максимальной прибыли
            if position.side == "LONG":
                current_profit = (current_price - position.entry_price) / position.entry_price
            else:
                current_profit = (position.entry_price - current_price) / position.entry_price
                
            position.max_profit = max(position.max_profit, current_profit)
            
            # Проверка на срабатывание стоп-лосса или тейк-профита
            if position.side == "LONG":
                if current_price <= position.stop_loss or current_price >= position.take_profit:
                    await self.close_position(symbol, current_price, "SL/TP triggered")
                    continue
                    
                # Обновление трейлинг-стопа
                trailing_distance = current_price * 0.02  # 2% трейлинг
                new_stop = current_price - trailing_distance
                
                if new_stop > position.trailing_stop:
                    await self._update_stop_loss(symbol, position, new_stop)
                    
                # Обновление трейлинг-тейк-профита при достижении 50% от цели
                if current_profit > 0.5 * ((position.take_profit - position.entry_price) / position.entry_price):
                    new_tp = current_price + (current_price - position.trailing_stop) * 1.5
                    if new_tp < position.trailing_take_profit:
                        await self._update_take_profit(symbol, position, new_tp)
                        
            elif position.side == "SHORT":
                if current_price >= position.stop_loss or current_price <= position.take_profit:
                    await self.close_position(symbol, current_price, "SL/TP triggered")
                    continue
                    
                # Обновление трейлинг-стопа
                trailing_distance = current_price * 0.02
                new_stop = current_price + trailing_distance
                
                if new_stop < position.trailing_stop:
                    await self._update_stop_loss(symbol, position, new_stop)
                    
                # Обновление трейлинг-тейк-профита
                if current_profit > 0.5 * ((position.entry_price - position.take_profit) / position.entry_price):
                    new_tp = current_price - (position.trailing_stop - current_price) * 1.5
                    if new_tp > position.trailing_take_profit:
                        await self._update_take_profit(symbol, position, new_tp)
    
    async def _update_stop_loss(self, symbol: str, position: Position, new_stop: float):
        """Обновление стоп-лосс ордера"""
        try:
            # Отмена старого ордера
            if symbol in self.stop_loss_orders:
                cancel_order(self.client, symbol, self.stop_loss_orders[symbol])
                
            # Размещение нового ордера
            close_side = "SELL" if position.side == "LONG" else "BUY"
            sl_order = place_stop_loss_order(
                self.client, symbol, close_side, position.quantity, new_stop
            )
            
            if sl_order:
                self.stop_loss_orders[symbol] = sl_order['orderId']
                position.trailing_stop = new_stop
                position.stop_loss = new_stop
                logger.info(f"Обновлен трейлинг-стоп для {symbol}: {new_stop:.8f}")
                
        except Exception as e:
            logger.error(f"Ошибка обновления стоп-лосса: {e}")
    
    async def _update_take_profit(self, symbol: str, position: Position, new_tp: float):
        """Обновление тейк-профит ордера"""
        try:
            # Отмена старого ордера
            if symbol in self.take_profit_orders:
                cancel_order(self.client, symbol, self.take_profit_orders[symbol])
                
            # Размещение нового ордера
            close_side = "SELL" if position.side == "LONG" else "BUY"
            tp_order = place_take_profit_order(
                self.client, symbol, close_side, position.quantity, new_tp
            )
            
            if tp_order:
                self.take_profit_orders[symbol] = tp_order['orderId']
                position.trailing_take_profit = new_tp
                position.take_profit = new_tp
                logger.info(f"Обновлен трейлинг-тейк-профит для {symbol}: {new_tp:.8f}")
                
        except Exception as e:
            logger.error(f"Ошибка обновления тейк-профита: {e}")
    
    async def close_position(self, symbol: str, current_price: float, reason: str = "") -> Optional[float]:
        """Закрытие позиции с записью в историю"""
        try:
            if symbol not in self.positions:
                logger.warning(f"Попытка закрыть несуществующую позицию: {symbol}")
                return None
                
            position = self.positions[symbol]
            
            # Отмена существующих ордеров
            if symbol in self.stop_loss_orders:
                cancel_order(self.client, symbol, self.stop_loss_orders[symbol])
                del self.stop_loss_orders[symbol]
                
            if symbol in self.take_profit_orders:
                cancel_order(self.client, symbol, self.take_profit_orders[symbol])
                del self.take_profit_orders[symbol]
                
            # Размещение рыночного ордера для закрытия позиции
            close_side = "SELL" if position.side == "LONG" else "BUY"
            order = place_market_order(self.client, symbol, close_side, position.quantity)
            
            if not order:
                logger.error(f"Не удалось закрыть позицию: {symbol}")
                return None
                
            # Расчет прибыли/убытка
            if position.side == "LONG":
                profit = position.quantity * (current_price - position.entry_price)
            else:
                profit = position.quantity * (position.entry_price - current_price)
                
            # Вычитаем комиссию
            profit -= position.quantity * current_price * COMMISSION * 2  # Открытие + закрытие
            
            # Запись в историю
            trade_record = {
                'symbol': symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'quantity': position.quantity,
                'profit': profit,
                'profit_percent': profit / (position.quantity * position.entry_price) * 100,
                'entry_time': position.entry_time,
                'exit_time': datetime.datetime.now(),
                'duration': (datetime.datetime.now() - position.entry_time).total_seconds() / 60,
                'market_regime': position.market_regime,
                'confidence_score': position.confidence_score,
                'max_profit': position.max_profit,
                'reason': reason
            }
            self.position_history.append(trade_record)
            logger.info(f"Закрыта позиция: {symbol} {position.side} с прибылью {profit:.2f} USDT "
                       f"({trade_record['profit_percent']:.2f}%), причина: {reason}")
            
            # Удаление позиции
            del self.positions[symbol]
            
            return profit
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции: {e}")
            logger.error(traceback.format_exc())
            return None

# --- Улучшенный торговый движок ---
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
        self.backtester = Backtester()
        self.commission = COMMISSION
        self.notification_manager = None

    # Добавить в EnhancedTradingEngine
    async def process_symbol(self, symbol: str):
        """Обрабатывает один символ: получает данные, делает предсказание, принимает решение"""
        try:
            model_data = self.models.get(symbol)
            if not model_data:
                logger.warning(f"[process_symbol] Нет модели для {symbol}")
                return

            model, scaler, features, performance = model_data

            # Загружаем последние 200 свечей
            df = fetch_binance_klines(self.client, symbol, TIMEFRAME, limit=200)
            if df.empty or len(df) < 50:
                logger.warning(f"[process_symbol] Недостаточно данных для {symbol}")
                return

            # Вычисляем индикаторы
            df = calculate_enhanced_indicators(df)
            if df.empty:
                logger.warning(f"[process_symbol] Не удалось рассчитать индикаторы для {symbol}")
                return

            market_conditions = self.market_detector.detect_regime(df)
            
            # Предсказание движения
            label, prob = predict_enhanced_direction(df, model, scaler, features, self.commission)
            confidence_score = abs(prob - 0.5) * 2

            # Генерация торгового сигнала
            signal, strength, reason, final_confidence = self.signal_analyzer.analyze_signal(
                df, prob, market_conditions, confidence_score
            )

            logger.info(f"[{symbol}] Сигнал: {signal}, Уверенность: {final_confidence:.2f}, Причина: {reason}")

            current_price = df['close'].iloc[-1]

            # Обработка сигнала и возможность открытия позиции
            if signal in ["BUY", "SELL"] and self.position_manager.can_open_position(symbol):
                atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.01
                side = "LONG" if signal == "BUY" else "SHORT"

                # Рассчитываем размер позиции и расстояние до стопа
                size, sl_distance = self.position_manager.calculate_dynamic_position_size(
                    symbol, current_price, atr, market_conditions, confidence_score
                )

                if size <= 0 or sl_distance <= 0:
                    logger.warning(f"[{symbol}] Нулевой размер позиции/стопа")
                    return

                stop_loss = current_price - sl_distance if signal == "BUY" else current_price + sl_distance
                take_profit = current_price + sl_distance * 2 if signal == "BUY" else current_price - sl_distance * 2

                # Открываем позицию
                position = await self.position_manager.open_position(
                    symbol, side, current_price, size,
                    stop_loss, take_profit, market_conditions, confidence_score
                )

                if position and self.notification_manager:
                    await self.notification_manager.send_trade_notification(position, "OPEN")

        except Exception as e:
            logger.error(f"[process_symbol] Ошибка обработки {symbol}: {e}")
            logger.error(traceback.format_exc())

    async def initialize_models(self):
        """Инициализация и обучение моделей с валидацией"""
        async with self._lock:
            try:
                klines_count = min(HISTORICAL_DAYS * 24 * 60 // 15, 1000)
                
                for symbol in SYMBOLS:
                    logger.info(f"Загрузка исторических данных для {symbol}")
                    df = fetch_binance_klines(self.client, symbol, TIMEFRAME, klines_count)
                    
                    if df.empty or len(df) < 500:
                        logger.error(f"Недостаточно данных для {symbol}")
                        continue
                        
                    logger.info(f"Обучение модели для {symbol}")
                    model, scaler, features = train_enhanced_model(df, self.commission)
                    
                    if model is None or scaler is None or not features:
                        logger.error(f"Не удалось обучить модель для {symbol}")
                        continue
                    
                    # Валидация модели через бэктест
                    logger.info(f"Валидация модели для {symbol}")
                    try:
                        df_backtest = df.copy()
                        df_backtest = calculate_enhanced_indicators(df_backtest)
                        
                        if df_backtest.empty:
                            logger.error(f"Не удалось рассчитать индикаторы для бэктеста {symbol}")
                            self.models[symbol] = (model, scaler, features, {'win_rate': 0.5, 'profit_factor': 1.0, 'total_trades': 0})
                            continue
                        
                        backtest_results = self._simple_backtest(df_backtest, model, scaler, features)
                        
                        logger.info(f"Результаты бэктеста для {symbol}: "
                                f"WR={backtest_results['win_rate']:.2%}, "
                                f"PF={backtest_results['profit_factor']:.2f}, "
                                f"Сделок={backtest_results['total_trades']}")
                        
                        self.models[symbol] = (model, scaler, features, backtest_results)
                        logger.info(f"Модель для {symbol} добавлена")
                            
                    except Exception as e:
                        logger.error(f"Ошибка валидации модели для {symbol}: {e}")
                        logger.error(traceback.format_exc())
                        self.models[symbol] = (model, scaler, features, {'win_rate': 0.5, 'profit_factor': 1.0, 'total_trades': 0})
                    
                if self.models:
                    logger.info(f"Успешно обучены модели для {len(self.models)} символов: {list(self.models.keys())}")
                    self.last_model_update = datetime.datetime.now()
                    
                    # Обновление корреляций
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

    def _simple_backtest(self, df: pd.DataFrame, model, scaler, features) -> dict:
        """Упрощенный бэктест для быстрой валидации модели"""
        try:
            logger.info("Запуск упрощенного бэктеста")
            
            min_lookback = 100
            if len(df) < min_lookback + 50:
                logger.warning("Недостаточно данных для бэктеста")
                return {'win_rate': 0.5, 'profit_factor': 1.0, 'total_trades': 0}
            
            trades = []
            
            for i in range(min_lookback, min(len(df), min_lookback + 500)):  # Увеличили до 500
                if i % 50 == 0:
                    logger.debug(f"Бэктест прогресс: {i}/{min(len(df), min_lookback + 500)}")
                    
                current_data = df.iloc[:i]
                
                # Подготовка признаков
                df_feat = prepare_enhanced_features(current_data, self.commission)
                if df_feat.empty or len(df_feat) < 10:
                    continue
                    
                # Проверка наличия всех признаков
                available_features = [f for f in features if f in df_feat.columns]
                if len(available_features) < len(features) * 0.8:
                    continue
                    
                # Заполнение отсутствующих признаков
                for f in features:
                    if f not in df_feat.columns:
                        df_feat[f] = 0
                        
                # Предсказание
                try:
                    X_latest = df_feat.iloc[-1][features].values.reshape(1, -1)
                    X_scaled = scaler.transform(X_latest)
                    prob = model.predict_proba(X_scaled)[0]
                    prob_up = prob[1]
                    
                    # Более строгие условия для входа
                    if prob_up > 0.65:  # Повысили порог
                        # Симулируем покупку
                        entry_price = df.iloc[i]['close']
                        
                        # Проверяем результат через несколько баров
                        max_profit = 0
                        for j in range(1, min(20, len(df) - i)):
                            exit_price = df.iloc[i + j]['close']
                            profit = (exit_price - entry_price) / entry_price
                            max_profit = max(max_profit, profit)
                            
                            # Выход по стоп-лоссу
                            if profit < -0.02:
                                trades.append(profit - self.commission * 2)
                                break
                            # Выход по тейк-профиту
                            elif profit > 0.03:
                                trades.append(profit - self.commission * 2)
                                break
                            # Выход по времени
                            elif j == 19:
                                trades.append(profit - self.commission * 2)
                                
                    elif prob_up < 0.35:  # Добавляем короткие позиции
                        # Симулируем продажу
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
            
            # Расчет метрик
            if trades:
                winning_trades = [t for t in trades if t > 0]
                win_rate = len(winning_trades) / len(trades)
                
                total_wins = sum(t for t in trades if t > 0)
                total_losses = abs(sum(t for t in trades if t < 0))
                profit_factor = total_wins / total_losses if total_losses > 0 else 1.0
                
                # Дополнительные метрики
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean([t for t in trades if t < 0]) if any(t < 0 for t in trades) else 0
                sharpe = np.mean(trades) / np.std(trades) * np.sqrt(252) if np.std(trades) > 0 else 0
                
                logger.info(f"Детали бэктеста: Avg Win: {avg_win:.4f}, Avg Loss: {avg_loss:.4f}, Sharpe: {sharpe:.2f}")
                
                return {
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_trades': len(trades),
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'sharpe_ratio': sharpe
                }
            else:
                return {'win_rate': 0.5, 'profit_factor': 1.0, 'total_trades': 0}
                
        except Exception as e:
            logger.error(f"Ошибка в упрощенном бэктесте: {e}")
            logger.error(traceback.format_exc())
            return {'win_rate': 0.5, 'profit_factor': 1.0, 'total_trades': 0}

# --- Улучшенная интеграция с Telegram ---
class EnhancedTelegramBot:
    def __init__(self):
        self.trading_engine = None
        self.application = None
        self.notification_manager = None
        self.trading_task = None
        
    async def setup(self):
        """Инициализация компонентов бота"""
        try:
            # Инициализация компонентов
            position_manager = EnhancedPositionManager(client)
            signal_analyzer = EnhancedSignalAnalyzer(COMMISSION, SPREAD)
            monitor = EnhancedPerformanceMonitor()
            self.trading_engine = EnhancedTradingEngine(client, position_manager, signal_analyzer, monitor)
            
            # Инициализация Telegram
            self.application = Application.builder().token(config['telegram']['token']).build()
            self.notification_manager = EnhancedNotificationManager(
                self.application.bot, 
                config['telegram']['chat_id']
            )
            
            # Установка notification manager в trading engine
            self.trading_engine.notification_manager = self.notification_manager
            
            # Регистрация обработчиков
            telegram_handler = EnhancedTelegramHandler(self.trading_engine, monitor)
            
            # Основные команды
            self.application.add_handler(CommandHandler("start", telegram_handler.handle_start))
            self.application.add_handler(CommandHandler("status", telegram_handler.handle_status))
            self.application.add_handler(CommandHandler("stats", telegram_handler.handle_stats))
            self.application.add_handler(CommandHandler("positions", telegram_handler.handle_positions))
            self.application.add_handler(CommandHandler("history", telegram_handler.handle_history))
            self.application.add_handler(CommandHandler("start_trading", telegram_handler.handle_start_trading))
            self.application.add_handler(CommandHandler("stop_trading", telegram_handler.handle_stop_trading))
            
            # Расширенные команды
            self.application.add_handler(CommandHandler("performance", telegram_handler.handle_performance))
            self.application.add_handler(CommandHandler("symbols", telegram_handler.handle_symbols))
            self.application.add_handler(CommandHandler("regimes", telegram_handler.handle_regimes))
            self.application.add_handler(CommandHandler("backtest", telegram_handler.handle_backtest))
            self.application.add_handler(CommandHandler("correlations", telegram_handler.handle_correlations))
            self.application.add_handler(CommandHandler("signals", telegram_handler.handle_signals))
            self.application.add_handler(CommandHandler("chart", telegram_handler.handle_chart))
            
            # Инициализация моделей
            await self.trading_engine.initialize_models()
            logger.info("Модели успешно инициализированы")
            
            # Отправка уведомления о запуске
            await self.notification_manager.send_startup_notification(self.trading_engine)
            
        except Exception as e:
            logger.error(f"Ошибка инициализации бота: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def start(self):
        """Запуск бота"""
        try:
            await self.setup()
            
            # Запуск торгового цикла
            self.trading_engine.is_running = True
            self.trading_task = asyncio.create_task(
                enhanced_trading_loop(self.trading_engine, self.notification_manager)
            )
            
            # Запуск Telegram бота
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            # Ждем завершения
            await asyncio.Event().wait()
            
        except Exception as e:
            logger.error(f"Ошибка запуска бота: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.stop()
    
    async def stop(self):
        """Остановка бота"""
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

# --- Улучшенный торговый цикл ---
async def enhanced_trading_loop(trading_engine: EnhancedTradingEngine, 
                              notification_manager: 'EnhancedNotificationManager'):
    """Основной торговый цикл с улучшенной логикой"""
    logger.info("Улучшенный торговый цикл запущен")
    
    last_performance_update = datetime.datetime.now()
    last_model_update = datetime.datetime.now()
    
    try:
        while trading_engine.is_running:
            try:
                current_time = datetime.datetime.now()
                
                # Проверка дневной статистики
                if (current_time - trading_engine.monitor.daily_stats['last_reset']).days > 0:
                    await notification_manager.send_daily_report(trading_engine.monitor)
                    trading_engine.monitor.reset_daily_stats()
                
                # Обновление моделей
                if (current_time - last_model_update).total_seconds() > 3600 * 6:
                    logger.info("Переобучение моделей...")
                    await trading_engine.initialize_models()
                    last_model_update = current_time
                
                # Обновление корреляций
                trading_engine.position_manager.correlation_analyzer.update_correlations(client)
                
                # Обновление позиций
                current_prices = {}
                for symbol in trading_engine.position_manager.positions.keys():
                    price = get_symbol_price(client, symbol)
                    if price:
                        current_prices[symbol] = price
                
                await trading_engine.position_manager.update_trailing_stops(current_prices)
                
                # Анализ символов
                tasks = []
                for symbol in SYMBOLS:
                    if symbol in trading_engine.models:
                        task = asyncio.create_task(trading_engine.process_symbol(symbol))
                        tasks.append(task)
                
                # Ждем завершения всех задач
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Периодическое обновление производительности
                if (current_time - last_performance_update).total_seconds() > 1800:
                    await notification_manager.send_performance_update(trading_engine.monitor)
                    last_performance_update = current_time
                
                await asyncio.sleep(UPDATE_INTERVAL_SEC)
                
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                logger.error(traceback.format_exc())
                await notification_manager.send_error_notification(f"Ошибка в торговом цикле: {e}")
                await asyncio.sleep(60)
                
    except asyncio.CancelledError:
        logger.info("Торговый цикл остановлен")
    finally:
        # Закрытие всех позиций
        for symbol in list(trading_engine.position_manager.positions.keys()):
            price = get_symbol_price(client, symbol)
            if price:
                await trading_engine.position_manager.close_position(symbol, price, "Shutdown")

# --- Улучшенные Telegram обработчики ---
class EnhancedTelegramHandler:
    def __init__(self, trading_engine: EnhancedTradingEngine, monitor: EnhancedPerformanceMonitor):
        self.trading_engine = trading_engine
        self.monitor = monitor
    
    async def handle_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детальный отчет о производительности"""
        try:
            metrics = self.monitor.performance_metrics
            
            # Создание графика эквити
            if self.monitor.trade_history:
                plt.figure(figsize=(10, 6))
                
                # График кумулятивной прибыли
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
        """Статистика по символам"""
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
        """Статистика по рыночным режимам"""
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
    
    async def handle_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Запуск бэктеста для символа"""
        try:
            args = context.args
            if not args:
                await update.message.reply_text("❌ Укажите символ: /backtest BTCUSDT")
                return
            
            symbol = args[0].upper()
            if symbol not in SYMBOLS:
                await update.message.reply_text(f"❌ Символ {symbol} не поддерживается")
                return
            
            await update.message.reply_text(f"⏳ Запуск бэктеста для {symbol}...")
            
            # Получение данных
            df = fetch_binance_klines(self.trading_engine.client, symbol, TIMEFRAME, 1000)
            if df.empty:
                await update.message.reply_text(f"❌ Не удалось получить данные для {symbol}")
                return
            
            # Запуск бэктеста
            if symbol in self.trading_engine.models:
                model, scaler, features, _ = self.trading_engine.models[symbol]
                
                backtester = Backtester()
                results = backtester.backtest_strategy(
                    df, model, scaler, features,
                    self.trading_engine.signal_analyzer,
                    self.trading_engine.market_detector,
                    self.trading_engine.threshold_manager
                )
                
                # Создание графика эквити
                plt.figure(figsize=(12, 8))
                
                # График 1: Кривая эквити
                plt.subplot(2, 1, 1)
                plt.plot(results['equity_curve'])
                plt.title(f'Кривая эквити - {symbol}')
                plt.xlabel('Время')
                plt.ylabel('Баланс (USDT)')
                plt.grid(True)
                
                # График 2: Просадка
                equity_array = np.array(results['equity_curve'])
                running_max = np.maximum.accumulate(equity_array)
                drawdown = (running_max - equity_array) / running_max * 100
                
                plt.subplot(2, 1, 2)
                plt.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
                plt.plot(drawdown, color='red')
                plt.title('Просадка (%)')
                plt.xlabel('Время')
                plt.ylabel('Просадка (%)')
                plt.grid(True)
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()
                
                # Формирование отчета
                caption = (
                    f"📊 <b>Результаты бэктеста {symbol}</b>\n\n"
                    f"💵 Начальный баланс: {backtester.initial_balance:.2f} USDT\n"
                    f"💰 Конечный баланс: {results['final_balance']:.2f} USDT\n"
                    f"📈 Общий доход: {results['total_return']:.2f}%\n"
                    f"📊 Всего сделок: {results['total_trades']}\n"
                    f"🎯 Win Rate: {results['win_rate']:.2%}\n"
                    f"💹 Profit Factor: {results['profit_factor']:.2f}\n"
                    f"📉 Max Drawdown: {results['max_drawdown']:.2f}%\n"
                    f"📈 Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
                    f"✅ Средний выигрыш: {results['avg_win']:.2f} USDT\n"
                    f"❌ Средний проигрыш: {results['avg_loss']:.2f} USDT\n"
                    f"🚀 Лучшая сделка: {results['best_trade']:.2f} USDT\n"
                    f"💥 Худшая сделка: {results['worst_trade']:.2f} USDT\n\n"
                    f"<b>Анализ по режимам:</b>\n"
                )
                
                for regime, stats in results['regime_analysis'].items():
                    caption += f"• {regime}: {stats['trades']} сделок, WR: {stats['win_rate']:.2%}\n"
                
                await update.message.reply_photo(photo=buf, caption=caption, parse_mode='HTML')
            else:
                await update.message.reply_text(f"❌ Модель для {symbol} не обучена")
                
        except Exception as e:
            logger.error(f"Ошибка бэктеста: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def handle_correlations(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Отображение корреляций между символами"""
        try:
            correlations = self.trading_engine.position_manager.correlation_analyzer.correlation_cache
            
            if correlations.empty:
                # Обновление корреляций
                correlations = self.trading_engine.position_manager.correlation_analyzer.update_correlations(
                    self.trading_engine.client
                )
            
            if correlations.empty:
                await update.message.reply_text("❌ Не удалось рассчитать корреляции")
                return
            
            # Создание тепловой карты
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Корреляционная матрица символов')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            
            # Поиск высоких корреляций
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
        """Текущие торговые сигналы для всех символов"""
        try:
            signals_msg = "📡 <b>Текущие торговые сигналы</b>\n\n"
            
            for symbol in SYMBOLS:
                if symbol not in self.trading_engine.models:
                    continue
                
                # Получение данных
                df = fetch_binance_klines(self.trading_engine.client, symbol, TIMEFRAME, 200)
                if df.empty:
                    continue
                
                df = calculate_enhanced_indicators(df)
                if df.empty:
                    continue
                
                # Анализ
                market_conditions = self.trading_engine.market_detector.detect_regime(df)
                model, scaler, features, _ = self.trading_engine.models[symbol]
                label, prob_up = predict_enhanced_direction(df, model, scaler, features, COMMISSION)
                
                confidence_score = abs(prob_up - 0.5) * 2
                signal, strength, reason, final_confidence = self.trading_engine.signal_analyzer.analyze_signal(
                    df, prob_up, market_conditions, confidence_score
                )
                
                # Эмодзи для сигнала
                signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
                
                # Текущая цена
                current_price = df['close'].iloc[-1]
                
                signals_msg += (
                    f"{signal_emoji} <b>{symbol}</b>: {signal}\n"
                    f"  💵 Цена: {current_price:.8f}\n"
                    f"  🤖 ML: {prob_up:.3f}\n"
                    f"  💪 Сила: {strength:.2f}\n"
                    f"  🌐 Режим: {market_conditions.regime}\n"
                    f"  📝 Причина: {reason}\n\n"
                )
            
            await update.message.reply_text(signals_msg, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Ошибка получения сигналов: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    async def handle_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Технический график для символа"""
        try:
            args = context.args
            if not args:
                await update.message.reply_text("❌ Укажите символ: /chart BTCUSDT")
                return
            
            symbol = args[0].upper()
            if symbol not in SYMBOLS:
                await update.message.reply_text(f"❌ Символ {symbol} не поддерживается")
                return
            
            # Получение данных
            df = fetch_binance_klines(self.trading_engine.client, symbol, TIMEFRAME, 200)
            if df.empty:
                await update.message.reply_text(f"❌ Не удалось получить данные для {symbol}")
                return
            
            df = calculate_enhanced_indicators(df)
            
            # Создание графиков
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            
            # График 1: Цена и MA
            ax1.plot(df.index, df['close'], label='Close', linewidth=2)
            if 'ma_20' in df:
                ax1.plot(df.index, df['ma_20'], label='MA20', alpha=0.7)
            if 'ma_50' in df:
                ax1.plot(df.index, df['ma_50'], label='MA50', alpha=0.7)
            if 'bb_upper' in df and 'bb_lower' in df:
                ax1.fill_between(df.index, df['bb_lower'], df['bb_upper'], alpha=0.2, color='gray')
            ax1.set_title(f'{symbol} - Цена и индикаторы')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Объем
            colors = ['green' if c > o else 'red' for c, o in zip(df['close'], df['open'])]
            ax2.bar(df.index, df['volume'], color=colors, alpha=0.7)
            ax2.set_title('Объем')
            ax2.grid(True, alpha=0.3)
            
            # График 3: RSI
            ax3.plot(df.index, df['rsi'], label='RSI', color='purple')
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
            ax3.set_title('RSI')
            ax3.set_ylim(0, 100)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # График 4: MACD
            ax4.plot(df.index, df['macd'], label='MACD', color='blue')
            ax4.plot(df.index, df['macd_signal'], label='Signal', color='red')
            ax4.bar(df.index, df['macd_hist'], label='Histogram', alpha=0.3)
            ax4.set_title('MACD')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close()
            
            # Анализ текущей ситуации
            market_conditions = self.trading_engine.market_detector.detect_regime(df)
            
            if symbol in self.trading_engine.models:
                model, scaler, features, _ = self.trading_engine.models[symbol]
                label, prob_up = predict_enhanced_direction(df, model, scaler, features, COMMISSION)
                confidence_score = abs(prob_up - 0.5) * 2
                signal, strength, reason, _ = self.trading_engine.signal_analyzer.analyze_signal(
                    df, prob_up, market_conditions, confidence_score
                )
            else:
                signal = "N/A"
                prob_up = 0.5
                reason = "Модель не обучена"
            
            caption = (
                f"📈 <b>{symbol} - Технический анализ</b>\n\n"
                f"💵 Цена: {df['close'].iloc[-1]:.8f}\n"
                f"📊 RSI: {df['rsi'].iloc[-1]:.2f}\n"
                f"📉 MACD: {df['macd_hist'].iloc[-1]:.8f}\n"
                f"📏 ATR: {df['atr'].iloc[-1]:.8f}\n"
                f"📊 Объем: {df['volume'].iloc[-1]:.2f}\n\n"
                f"🔍 <b>Анализ:</b>\n"
                f"Режим рынка: {market_conditions.regime}\n"
                f"ML прогноз: {prob_up:.3f}\n"
                f"Сигнал: {signal}\n"
                f"Причина: {reason}"
            )
            
            await update.message.reply_photo(photo=buf, caption=caption, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Ошибка создания графика: {e}")
            await update.message.reply_text(f"❌ Ошибка: {e}")
    
    # Базовые обработчики
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = (
                        "🤖 <b>Улучшенный криптовалютный торговый бот v2.0</b>\n\n"
            "🚀 <b>Новые функции:</b>\n"
            "• Ансамблевые ML модели (RF + XGBoost)\n"
            "• Адаптивное управление рисками\n"
            "• Детектор рыночных режимов\n"
            "• Анализ корреляций\n"
            "• Трейлинг тейк-профит\n"
            "• Расширенная статистика\n\n"
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
            "/backtest SYMBOL - Бэктест\n"
            "/start_trading - Запуск торговли\n"
            "/stop_trading - Остановка торговли"
        )
        await update.message.reply_text(welcome_message, parse_mode='HTML')
    
    async def handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /status"""
        try:
            balance = get_usdt_balance(self.trading_engine.client)
            active_positions = len(self.trading_engine.position_manager.positions)
            daily_profit = self.monitor.daily_stats['profit']
            
            # Текущие позиции
            positions_info = ""
            if self.trading_engine.position_manager.positions:
                positions_info = "\n\n<b>Активные позиции:</b>\n"
                for symbol, pos in self.trading_engine.position_manager.positions.items():
                    positions_info += f"• {symbol} {pos.side} ({pos.confidence_score:.2f})\n"
            
            status_msg = (
                "📊 <b>Текущий статус</b>\n\n"
                f"💰 Баланс: {balance:.2f} USDT\n"
                f"📈 Дневной профит: {daily_profit:.2f} USDT ({daily_profit/balance*100:.2f}%)\n"
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
        """Обработчик команды /stats"""
        try:
            metrics = self.monitor.performance_metrics
            
            # Лучшие часы для торговли
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
        """Обработчик команды /positions"""
        try:
            if not self.trading_engine.position_manager.positions:
                await update.message.reply_text("📊 Нет активных позиций")
                return
            
            positions_msg = "💼 <b>Активные позиции</b>\n\n"
            
            for symbol, pos in self.trading_engine.position_manager.positions.items():
                current_price = get_symbol_price(self.trading_engine.client, symbol)
                if not current_price:
                    continue
                
                # Расчет текущей прибыли
                if pos.side == "LONG":
                    current_profit = pos.quantity * (current_price - pos.entry_price)
                    profit_percent = (current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    current_profit = pos.quantity * (pos.entry_price - current_price)
                    profit_percent = (pos.entry_price - current_price) / pos.entry_price * 100
                
                # Эмодзи для прибыли
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
        """Обработчик команды /history"""
        try:
            if not self.monitor.trade_history:
                await update.message.reply_text("📊 История торговли пуста")
                return
            
            # Последние 10 сделок
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
        """Обработчик команды /start_trading"""
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
        """Обработчик команды /stop_trading"""
        try:
            if not self.trading_engine.is_running:
                await update.message.reply_text("🔴 Бот уже остановлен")
                return
            
            self.trading_engine.is_running = False
            
            # Закрытие всех позиций
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

# --- Улучшенный менеджер уведомлений ---
class EnhancedNotificationManager:
    def __init__(self, bot, chat_id):
        self.bot = bot
        self.chat_id = chat_id
        self.last_notification = {}
        self.notification_cooldown = 300  # 5 минут
    
    async def send_notification(self, message: str, parse_mode: str = 'HTML', 
                               notification_type: str = 'general'):
        """Отправка уведомления с защитой от спама"""
        try:
            # Проверка cooldown
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
            
            # Лучшие символы дня
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

# --- Главная функция запуска ---
async def main():
    """Основная функция запуска улучшенного бота"""
    try:
        # Инициализация компонентов
        position_manager = EnhancedPositionManager(client)
        signal_analyzer = EnhancedSignalAnalyzer(COMMISSION, SPREAD)
        monitor = EnhancedPerformanceMonitor()
        trading_engine = EnhancedTradingEngine(client, position_manager, signal_analyzer, monitor)
        
        # Инициализация моделей
        logger.info("Инициализация и валидация моделей...")
        await trading_engine.initialize_models()
        
        # Запуск торгового цикла
        logger.info("Запуск торгового цикла...")
        trading_engine.is_running = True
        
        while trading_engine.is_running:
            try:
                # Обновление моделей при необходимости
                if (datetime.datetime.now() - trading_engine.last_model_update).total_seconds() > 3600 * 6:
                    logger.info("Обновление моделей...")
                    await trading_engine.initialize_models()
                
                # Обновление корреляций
                if (datetime.datetime.now() - position_manager.correlation_analyzer.last_update).total_seconds() > 3600:
                    position_manager.correlation_analyzer.update_correlations(client)
                
                # Обновление открытых позиций
                current_prices = {}
                for symbol in position_manager.positions.keys():
                    price = get_symbol_price(client, symbol)
                    if price:
                        current_prices[symbol] = price
                
                await position_manager.update_trailing_stops(current_prices)
                
                # Обработка символов
                for symbol in SYMBOLS:
                    if symbol in trading_engine.models:
                        await trading_engine.process_symbol(symbol)
                        await asyncio.sleep(1)  # Небольшая задержка между символами
                
                # Периодический отчет о производительности
                if datetime.datetime.now().minute % 30 == 0:
                    logger.info(f"=== Отчет о производительности ===")
                    logger.info(f"Открытых позиций: {len(position_manager.positions)}")
                    logger.info(f"Общая прибыль: {monitor.performance_metrics.total_profit:.2f} USDT")
                    logger.info(f"Win Rate: {monitor.performance_metrics.win_rate:.2%}")
                    logger.info(f"Profit Factor: {monitor.performance_metrics.profit_factor:.2f}")
                    
                    # Лучшие символы
                    best_symbols = monitor.get_best_symbols(3)
                    if best_symbols:
                        logger.info("Лучшие символы:")
                        for symbol, stats in best_symbols:
                            logger.info(f"  {symbol}: {stats['profit']:.2f} USDT, WR: {stats['win_rate']:.2%}")
                    
                    # Производительность по режимам
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
        # Закрытие всех позиций
        logger.info("Закрытие всех открытых позиций...")
        for symbol in list(position_manager.positions.keys()):
            price = get_symbol_price(client, symbol)
            if price:
                await position_manager.close_position(symbol, price, "Shutdown")
        
        logger.info("Бот остановлен")

# --- Точка входа ---
if __name__ == "__main__":
    try:
        # Проверка конфигурации
        if not config.get('telegram', {}).get('token'):
            logger.error("Telegram токен не настроен")
            sys.exit(1)
        
        if not config.get('binance', {}).get('api_key'):
            logger.error("Binance API ключи не настроены")
            sys.exit(1)
        
        # Выбор режима запуска
        if len(sys.argv) > 1 and sys.argv[1] == "--no-telegram":
            # Запуск без Telegram
            logger.info("Запуск в режиме без Telegram...")
            asyncio.run(main())
        else:
            # Запуск с Telegram
            logger.info("Запуск с Telegram интеграцией...")
            bot = EnhancedTelegramBot()
            asyncio.run(bot.start())
        
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        logger.error(traceback.format_exc())