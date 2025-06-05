import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
import datetime
import talib
from typing import Any, Dict
from collections import deque

logger = logging.getLogger("calculations")

# --- Структуры данных (оставляем как есть) ---
@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: float
    side: str  # "LONG" или "SHORT"
    stop_loss: float
    take_profit: float
    trailing_stop: float
    trailing_take_profit: float
    entry_time: datetime.datetime
    order_id: str = None
    market_regime: str = None
    confidence_score: float = 0.0
    max_profit: float = 0.0
    entry_slippage: float = 0.0
    entry_commission: float = 0.0
    _closing: bool = field(default=False, init=False)
    partially_closed: bool = field(default=False, init=False)

@dataclass
class TradeStats:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

@dataclass
class MarketConditions:
    regime: str  # trending_up, trending_down, ranging, high_volatility
    volatility_percentile: float
    volume_percentile: float
    trend_strength: float
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)

# --- Расширенные технические индикаторы ---
class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Расчет индикатора RSI с безопасным делением"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = safe_divide(gain, loss, 0)
        return 100 - safe_divide(100, 1 + rs, 0)

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Расчет индикатора ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет индикатора MACD"""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет полос Боллинджера"""
        middle_band = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Расчет стохастического осциллятора"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Расчет VWAP"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Расчет On Balance Volume"""
        return (volume * np.sign(close.diff())).cumsum()

    @staticmethod
    def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                 volume: pd.Series, period: int = 14) -> pd.Series:
        """Расчет Money Flow Index с безопасным делением"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()
        
        mfi_ratio = safe_divide(positive_flow_sum, negative_flow_sum, 0)
        return 100 - safe_divide(100, 1 + mfi_ratio, 0)
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Расчет ADX (Average Directional Index)"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = TechnicalIndicators.calculate_atr(high, low, close, 1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Расчет Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Расчет Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    
    @staticmethod
    def calculate_ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                                   period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
        """Расчет Ultimate Oscillator"""
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([high - low, 
                       abs(high - close.shift(1)), 
                       abs(low - close.shift(1))], axis=1).max(axis=1)
        
        avg1 = bp.rolling(period1).sum() / tr.rolling(period1).sum()
        avg2 = bp.rolling(period2).sum() / tr.rolling(period2).sum()
        avg3 = bp.rolling(period3).sum() / tr.rolling(period3).sum()
        
        uo = 100 * ((4 * avg1 + 2 * avg2 + avg3) / 7)
        return uo
    
    @staticmethod
    def calculate_roc(series: pd.Series, period: int = 10) -> pd.Series:
        """Расчет Rate of Change"""
        return ((series - series.shift(period)) / series.shift(period)) * 100
    
    @staticmethod
    def calculate_ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Расчет Percentage Price Oscillator"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        ppo_signal = ppo.ewm(span=signal).mean()
        return ppo, ppo_signal
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Расчет индикатора ATR с обработкой ошибок"""
        try:
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            # Объединяем и берем максимум
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR - экспоненциальное скользящее среднее True Range
            atr = tr.ewm(span=period, adjust=False).mean()
            
            return atr
        except Exception as e:
            logger.error(f"Ошибка расчета ATR: {e}")
            return pd.Series(index=high.index, dtype=float)


class IndicatorCache:
    """Класс для кэширования расчетов индикаторов"""
    def __init__(self):
        self.cache = {}
        
    def get_key(self, df_length: int, last_close: float) -> str:
        return f"{df_length}_{last_close}"
    
    def get(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        key = self.get_key(len(df), df['close'].iloc[-1])
        return self.cache.get(key)
    
    def set(self, df: pd.DataFrame, result: pd.DataFrame):
        key = self.get_key(len(df), df['close'].iloc[-1])
        self.cache[key] = result.copy()
        
        # Ограничиваем размер кэша
        if len(self.cache) > 100:
            # Удаляем старые записи
            keys = list(self.cache.keys())
            for k in keys[:20]:
                del self.cache[k]

# Глобальный экземпляр кэша
_indicator_cache = IndicatorCache()

# --- Детектор рыночных режимов с улучшенной логикой ---
class MarketRegimeDetector:
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        
    def detect_regime(self, df: pd.DataFrame) -> MarketConditions:
        """Определение текущего рыночного режима"""
        try:
            # Убедимся, что все необходимые индикаторы рассчитаны
            if 'atr' not in df.columns:
                ti = TechnicalIndicators()
                df['atr'] = ti.calculate_atr(df['high'], df['low'], df['close'])
            
            if 'adx' not in df.columns:
                df['adx'] = TechnicalIndicators.calculate_adx(df['high'], df['low'], df['close'])
            
            # Расчет волатильности
            current_vol = df['atr'].iloc[-1] / df['close'].iloc[-1]
            vol_series = df['atr'] / df['close']
            vol_percentile = (vol_series < current_vol).sum() / len(vol_series) * 100
            
            # Расчет объема
            current_volume = df['volume'].iloc[-1]
            volume_percentile = (df['volume'] < current_volume).sum() / len(df['volume']) * 100
            
            # Определение тренда
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) > 200 else sma_50
            
            # ADX для силы тренда
            trend_strength = df['adx'].iloc[-1] if 'adx' in df.columns else 25
            
            # Определение уровней поддержки и сопротивления
            support_levels, resistance_levels = self._find_support_resistance(df)
            
            # Определение режима с улучшенной логикой
            if vol_percentile > 80:
                regime = "high_volatility"
            elif trend_strength > 25:
                if sma_20 > sma_50 > sma_200:
                    regime = "trending_up"
                elif sma_20 < sma_50 < sma_200:
                    regime = "trending_down"
                else:
                    regime = "ranging"
            else:
                regime = "ranging"
                
            return MarketConditions(
                regime=regime,
                volatility_percentile=vol_percentile,
                volume_percentile=volume_percentile,
                trend_strength=trend_strength,
                support_levels=support_levels,
                resistance_levels=resistance_levels
            )
            
        except Exception as e:
            logger.error(f"Ошибка определения рыночного режима: {e}")
            return MarketConditions(
                regime="unknown",
                volatility_percentile=50,
                volume_percentile=50,
                trend_strength=0
            )
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
        """Поиск уровней поддержки и сопротивления с улучшенным алгоритмом"""
        # Поиск локальных минимумов и максимумов
        highs = df['high'].rolling(window, center=True).max() == df['high']
        lows = df['low'].rolling(window, center=True).min() == df['low']
        
        current_price = df['close'].iloc[-1]
        
        # Фильтрация значимых уровней
        resistance_candidates = df.loc[highs, 'high'].unique()
        support_candidates = df.loc[lows, 'low'].unique()
        
        # Кластеризация близких уровней
        def cluster_levels(levels, threshold=0.01):
            if len(levels) == 0:
                return []
            
            sorted_levels = sorted(levels)
            clusters = [[sorted_levels[0]]]
            
            for level in sorted_levels[1:]:
                if abs(level - clusters[-1][-1]) / clusters[-1][-1] < threshold:
                    clusters[-1].append(level)
                else:
                    clusters.append([level])
            
            return [np.mean(cluster) for cluster in clusters]
        
        resistance_levels = cluster_levels([r for r in resistance_candidates if r > current_price])[:3]
        support_levels = cluster_levels([s for s in support_candidates if s < current_price])[-3:]
        
        return support_levels, resistance_levels

# --- Оптимизатор стоп-лоссов с улучшенной логикой ---
class StopLossOptimizer:
    def __init__(self):
        self.atr_multipliers = {
                        "high_volatility": 3.0,
            "trending_up": 2.0,
            "trending_down": 2.0,
            "ranging": 2.5,
            "unknown": 2.5
        }
        
    def optimize_stop_loss(self, df: pd.DataFrame, position_side: str, 
                          market_conditions: MarketConditions) -> Tuple[float, float]:
        """Оптимизация стоп-лосса и тейк-профита на основе рыночных условий"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Базовый множитель ATR
        atr_multiplier = self.atr_multipliers.get(market_conditions.regime, 2.5)
        
        # Корректировка на основе волатильности
        if market_conditions.volatility_percentile > 80:
            atr_multiplier *= 1.3
        elif market_conditions.volatility_percentile < 20:
            atr_multiplier *= 0.7
            
        # Расчет на основе структуры рынка
        if position_side == "LONG":
            # Использование уровней поддержки для стоп-лосса
            if market_conditions.support_levels:
                nearest_support = max([s for s in market_conditions.support_levels if s < current_price], 
                                    default=current_price - atr * atr_multiplier)
                stop_loss = max(nearest_support * 0.995, current_price - atr * atr_multiplier)
            else:
                stop_loss = current_price - atr * atr_multiplier
                
            # Использование уровней сопротивления для тейк-профита
            if market_conditions.resistance_levels:
                nearest_resistance = min([r for r in market_conditions.resistance_levels if r > current_price], 
                                       default=current_price + atr * atr_multiplier * 3)
                take_profit = min(nearest_resistance * 0.995, current_price + atr * atr_multiplier * 3)
            else:
                take_profit = current_price + atr * atr_multiplier * 3
                
        else:  # SHORT
            if market_conditions.resistance_levels:
                nearest_resistance = min([r for r in market_conditions.resistance_levels if r > current_price], 
                                       default=current_price + atr * atr_multiplier)
                stop_loss = min(nearest_resistance * 1.005, current_price + atr * atr_multiplier)
            else:
                stop_loss = current_price + atr * atr_multiplier
                
            if market_conditions.support_levels:
                nearest_support = max([s for s in market_conditions.support_levels if s < current_price], 
                                    default=current_price - atr * atr_multiplier * 3)
                take_profit = max(nearest_support * 1.005, current_price - atr * atr_multiplier * 3)
            else:
                take_profit = current_price - atr * atr_multiplier * 3
                
        return stop_loss, take_profit


def safe_divide(numerator, denominator, default=0):
    """Безопасное деление с обработкой деления на ноль"""
    if isinstance(denominator, (pd.Series, np.ndarray)):
        return np.where(denominator != 0, numerator / denominator, default)
    else:
        return numerator / denominator if denominator != 0 else default

def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Расчет всех технических индикаторов с безопасным делением"""
    try:
        if df.empty or len(df) < 50:
            logger.warning("Недостаточно данных для расчета индикаторов")
            return pd.DataFrame()
        
        df = df.copy()
        ti = TechnicalIndicators()
        
        # Проверяем наличие необходимых колонок
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Отсутствуют необходимые колонки: {missing_columns}")
            return pd.DataFrame()
        
        # Базовые расчеты
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(safe_divide(df['close'], df['close'].shift(1), 1))
        
        # RSI
        df['rsi'] = ti.calculate_rsi(df['close'])
        df['rsi_ma'] = df['rsi'].rolling(10).mean()
        
        # ATR
        df['atr'] = ti.calculate_atr(df['high'], df['low'], df['close'])
        df['atr_norm'] = safe_divide(df['atr'], df['close'])
        
        # MACD
        macd_line, signal_line, hist = ti.calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = hist
        df['macd_hist_slope'] = df['macd_hist'].diff()
        
        # Bollinger Bands
        upper_band, middle_band, lower_band = ti.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper_band
        df['bb_middle'] = middle_band
        df['bb_lower'] = lower_band
        df['bb_width'] = safe_divide(df['bb_upper'] - df['bb_lower'], df['bb_middle'])
        df['bb_position'] = safe_divide(df['close'] - df['bb_lower'], df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        k_percent, d_percent = ti.calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
        
        # Volume indicators
        df['obv'] = ti.calculate_obv(df['close'], df['volume'])
        df['obv_ma'] = df['obv'].rolling(10).mean()
        df['obv_slope'] = df['obv'].pct_change(5)
        
        df['mfi'] = ti.calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
        df['vwap'] = ti.calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
        df['price_to_vwap'] = safe_divide(df['close'], df['vwap'], 1)
        
        # Дополнительные индикаторы
        df['adx'] = ti.calculate_adx(df['high'], df['low'], df['close'])
        df['cci'] = ti.calculate_cci(df['high'], df['low'], df['close'])
        df['williams_r'] = ti.calculate_williams_r(df['high'], df['low'], df['close'])
        df['ultimate_osc'] = ti.calculate_ultimate_oscillator(df['high'], df['low'], df['close'])
        df['roc'] = ti.calculate_roc(df['close'])
        
        ppo, ppo_signal = ti.calculate_ppo(df['close'])
        df['ppo'] = ppo
        df['ppo_signal'] = ppo_signal
        df['ppo_hist'] = df['ppo'] - df['ppo_signal']
        
        # Скользящие средние
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                df[f'ma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                df[f'ma_{period}_slope'] = df[f'ma_{period}'].pct_change(5)
        
        # Добавляем MA 100 и 200 если достаточно данных
        for period in [100, 200]:
            if len(df) >= period:
                df[f'ma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = safe_divide(df['volume'], df['volume_ma'])
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std()
        df['volatility_ratio'] = safe_divide(df['volatility'], df['close'])
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio_20'] = safe_divide(df['volatility_20'], df['volatility_20'].rolling(100).mean())
        
        # Лаги и изменения
        for lag in range(1, 9):
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Доходности за разные периоды
        for period in [1, 5, 10, 20, 60]:
            df[f'return_{period}m'] = df['close'].pct_change(period)
        
        # Изменение объема
        df['volume_change_5m'] = df['volume'].pct_change(5)
        
        # Отношение цены к скользящим средним
        if 'ema_20' in df.columns:
            df['ema_20_slope'] = df['ema_20'].pct_change(5)
        if 'ema_50' in df.columns:
            df['price_to_ema_50'] = safe_divide(df['close'], df['ema_50'], 1)
        
        # Расстояние между MA
        if 'ma_20' in df.columns and 'ma_50' in df.columns:
            df['ma_cross_20_50_distance'] = safe_divide(df['ma_20'] - df['ma_50'], df['close'])
        
        # Временные признаки
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
        else:
            try:
                df['hour'] = pd.to_datetime(df.index).hour
                df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            except:
                df['hour'] = np.arange(len(df)) % 24
                df['day_of_week'] = (np.arange(len(df)) // 24) % 7
        
        # Циклические преобразования времени
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Сила тренда
        df['trend_strength'] = df['adx'] / 100  # Нормализуем ADX
        
        # Моментум
        df['momentum_20'] = safe_divide(df['close'], df['close'].shift(20), 1) - 1
        df['momentum_50'] = safe_divide(df['close'], df['close'].shift(50), 1) - 1
        
        # Статистические метрики доходности
        for window in [10, 20, 50]:
            if len(df) >= window:
                returns = df['returns']
                df[f'return_mean_{window}'] = returns.rolling(window).mean()
                df[f'return_std_{window}'] = returns.rolling(window).std()
                df[f'return_skew_{window}'] = returns.rolling(window).skew()
                df[f'return_kurt_{window}'] = returns.rolling(window).kurt()
        
        # Volume Price Trend
        df['volume_price_trend'] = (df['volume'] * df['close'].pct_change()).cumsum()
        
        # Accumulation/Distribution
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow_multiplier = safe_divide(
            (df['close'] - df['low']) - (df['high'] - df['close']), 
            df['high'] - df['low']
        )
        money_flow_volume = money_flow_multiplier * df['volume']
        df['accumulation_distribution'] = money_flow_volume.cumsum()
        
        # RSI-MFI дивергенция
        if 'rsi' in df.columns and 'mfi' in df.columns:
            df['rsi_mfi_divergence'] = df['rsi'] - df['mfi']
        
        # Расстояние от максимума/минимума
        df['distance_from_high'] = safe_divide(df['close'], df['high'].rolling(20).max(), 1) - 1
        df['distance_from_low'] = safe_divide(df['close'], df['low'].rolling(20).min(), 1) - 1
        
        # Фрактальная размерность
        def fractal_dimension(series, window=20):
            if len(series) < window:
                return np.nan
            range_price = series.max() - series.min()
            if range_price == 0:
                return 1.0
            line_length = np.sum(np.abs(np.diff(series)))
            return safe_divide(np.log(line_length + 1), np.log(range_price + 1), 1)
        
        df['fractal_dimension'] = df['close'].rolling(20).apply(fractal_dimension, raw=True)
        
        # Свечные паттерны
        df['hammer'] = ((df['high'] - df['low']) > 3 * abs(df['close'] - df['open'])) & \
                       (safe_divide(df['close'] - df['low'], df['high'] - df['low']) > 0.6) & \
                       (safe_divide(df['open'] - df['low'], df['high'] - df['low']) > 0.6)
        df['hammer'] = df['hammer'].astype(int)
        
        df['shooting_star'] = ((df['high'] - df['low']) > 3 * abs(df['close'] - df['open'])) & \
                              (safe_divide(df['high'] - df['close'], df['high'] - df['low']) > 0.6) & \
                              (safe_divide(df['high'] - df['open'], df['high'] - df['low']) > 0.6)
        df['shooting_star'] = df['shooting_star'].astype(int)
        
        # Удаляем строки с NaN в начале (из-за rolling операций)
        initial_nans = df.isna().all(axis=1).sum()
        if initial_nans > 0:
            df = df.iloc[initial_nans:]
        
        # Финальная очистка
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Проверяем, что DataFrame не пустой
        if df.empty:
            logger.warning("DataFrame пустой после расчета индикаторов")
            return pd.DataFrame()
        
        logger.debug(f"Рассчитано {len(df.columns)} индикаторов для {len(df)} строк")
        
        return df
        
    except Exception as e:
        import traceback
        logger.error(f"Ошибка расчета индикаторов: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


# --- Анализатор корреляций ---
class CorrelationAnalyzer:
    def __init__(self, symbols: List[str], correlation_window: int = 100):
        self.symbols = symbols
        self.correlation_window = correlation_window
        self.correlation_cache = pd.DataFrame()
        self.last_update = datetime.datetime.now()  # Инициализируем текущим временем

        
    def update_correlations(self, client) -> pd.DataFrame:
        """Обновление корреляционной матрицы"""
        try:
            logger.debug(f"Начало обновления корреляций для {len(self.symbols)} символов")
            prices = {}
            
            for symbol in self.symbols:
                try:
                    logger.debug(f"Получение данных для корреляций: {symbol}")
                    from binance_utils import fetch_binance_klines
                    from __main__ import TIMEFRAME
                    df = fetch_binance_klines(client, symbol, TIMEFRAME, 1000)
                    if not df.empty and len(df) >= 50:
                        prices[symbol] = df['close']
                    else:
                        logger.warning(f"Недостаточно данных для {symbol} в корреляциях")
                except Exception as e:
                    logger.error(f"Ошибка получения данных для {symbol}: {e}")
                    continue
            
            if len(prices) < 2:
                logger.warning("Недостаточно данных для расчета корреляций")
                return pd.DataFrame()
                
            price_df = pd.DataFrame(prices)
            price_df = price_df.dropna()
            
            if len(price_df) < 20:
                logger.warning("Недостаточно данных после очистки для корреляций")
                return pd.DataFrame()
                
            # Расчет корреляций на основе доходностей
            returns_df = price_df.pct_change().dropna()
            correlations = returns_df.corr()
            
            self.correlation_cache = correlations
            self.last_update = datetime.datetime.now()
            
            logger.info(f"Корреляции обновлены для {len(correlations)} символов")
            return correlations
            
        except Exception as e:
            logger.error(f"Ошибка расчета корреляций: {e}")
            return pd.DataFrame()
    
    def should_trade_symbol(self, symbol: str, open_positions: Dict[str, Position], 
                          max_correlation: float = 0.7) -> bool:
        """Проверка возможности торговли символом с учетом корреляций"""
        if not open_positions or symbol not in self.correlation_cache.index:
            return True
            
        for pos_symbol in open_positions:
            if pos_symbol in self.correlation_cache.index and symbol in self.correlation_cache.columns:
                corr = abs(self.correlation_cache.loc[pos_symbol, symbol])
                if corr > max_correlation:
                    logger.info(f"Высокая корреляция {corr:.2f} между {symbol} и {pos_symbol}")
                    return False
        return True

# --- Функция для получения дополнительных данных ---
def fetch_additional_market_data(client, symbol: str) -> Dict[str, Any]:
    """Получение дополнительных рыночных данных"""
    try:
        # Получение данных о 24-часовой статистике
        ticker_24h = client.get_ticker(symbol=symbol)
        
        # Получение стакана ордеров
        order_book = client.get_order_book(symbol=symbol, limit=20)
        
        # Расчет дополнительных метрик
        bid_volume = sum(float(bid[1]) for bid in order_book['bids'][:5])
        ask_volume = sum(float(ask[1]) for ask in order_book['asks'][:5])
        
        bid_ask_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
        
        # Спред
        best_bid = float(order_book['bids'][0][0]) if order_book['bids'] else 0
        best_ask = float(order_book['asks'][0][0]) if order_book['asks'] else 0
        spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
        
        return {
            'price_change_24h': float(ticker_24h['priceChangePercent']),
            'volume_24h': float(ticker_24h['volume']),
            'quote_volume_24h': float(ticker_24h['quoteVolume']),
            'count_24h': int(ticker_24h['count']),
            'weighted_avg_price': float(ticker_24h['weightedAvgPrice']),
            'bid_ask_imbalance': bid_ask_imbalance,
            'spread': spread,
            'bid_volume_5': bid_volume,
            'ask_volume_5': ask_volume
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения дополнительных данных для {symbol}: {e}")
        return {}

# MarketMicrostructureAnalyzer

class MarketMicrostructureAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.order_book_history = {}
        self.trade_history = {}
        
    def update_order_book(self, symbol: str, order_book_data: dict):
        """Обновление истории стакана ордеров"""
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = deque(maxlen=self.window_size)
        
        self.order_book_history[symbol].append({
            'timestamp': datetime.datetime.now(),
            'data': order_book_data
        })
    
    def update_trades(self, symbol: str, trades_df: pd.DataFrame):
        """Обновление истории сделок"""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=self.window_size * 10)
        
        for _, trade in trades_df.iterrows():
            self.trade_history[symbol].append(trade)
    
    def calculate_order_flow_metrics(self, symbol: str) -> dict:
        """Расчет метрик потока ордеров"""
        if symbol not in self.order_book_history or not self.order_book_history[symbol]:
            return {}
            
        # Последние данные стакана
        latest_book = self.order_book_history[symbol][-1]['data']
        
        # Изменение дисбаланса
        imbalance_history = [entry['data'].get('imbalance', 0) for entry in self.order_book_history[symbol]]
        imbalance_change = np.diff(imbalance_history[-10:]) if len(imbalance_history) >= 10 else []
        
        # Изменение спреда
        spread_history = [entry['data'].get('spread_pct', 0) for entry in self.order_book_history[symbol]]
        spread_change = np.mean(np.diff(spread_history[-10:])) if len(spread_history) >= 10 else 0
        
        # Метрики давления
        buy_pressure_history = [entry['data'].get('buy_pressure', 1) for entry in self.order_book_history[symbol]]
        sell_pressure_history = [entry['data'].get('sell_pressure', 1) for entry in self.order_book_history[symbol]]
        
        avg_buy_pressure = np.mean(buy_pressure_history[-10:]) if len(buy_pressure_history) >= 10 else 1
        avg_sell_pressure = np.mean(sell_pressure_history[-10:]) if len(sell_pressure_history) >= 10 else 1
        
        # Расчет кумулятивного дельта-объема (CVD)
        cvd = 0
        if symbol in self.trade_history and self.trade_history[symbol]:
            trades = list(self.trade_history[symbol])
            buy_volume = sum(trade.get('buy_volume', 0) for trade in trades)
            sell_volume = sum(trade.get('sell_volume', 0) for trade in trades)
            cvd = buy_volume - sell_volume
        
        return {
            'imbalance': latest_book.get('imbalance', 0),
            'imbalance_trend': np.mean(imbalance_change) if len(imbalance_change) > 0 else 0,
            'spread_pct': latest_book.get('spread_pct', 0),
            'spread_trend': spread_change,
            'buy_pressure': avg_buy_pressure,
            'sell_pressure': avg_sell_pressure,
            'pressure_ratio': avg_buy_pressure / avg_sell_pressure if avg_sell_pressure > 0 else 1,
            'cvd': cvd,
            'cvd_normalized': cvd / sum(trade.get('qty', 0) for trade in self.trade_history[symbol]) if self.trade_history.get(symbol) else 0
        }
    
    def detect_price_jumps(self, symbol: str, threshold: float = 0.003) -> bool:
        """Обнаружение скачков цены"""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < 20:
            return False
            
        trades = list(self.trade_history[symbol])
        prices = [float(trade.get('price', 0)) for trade in trades]
        
        # Расчет возвратов
        returns = np.diff(prices) / prices[:-1]
        
        # Обнаружение скачков (возврат больше порога)
        jumps = [abs(ret) > threshold for ret in returns]
        
        return any(jumps[-10:])  # Проверка последних 10 возвратов
    
    def calculate_market_impact(self, symbol: str, order_size: float) -> float:
        """Оценка рыночного влияния для заданного размера ордера"""
        if symbol not in self.order_book_history or not self.order_book_history[symbol]:
            return 0.0
            
        latest_book = self.order_book_history[symbol][-1]['data']
        asks = latest_book.get('asks', [])
        
        if not asks:
            return 0.0
            
        remaining_size = order_size
        weighted_price = 0
        base_price = asks[0]['price']
        
        for ask in asks:
            if remaining_size <= 0:
                break
                
            executed_qty = min(remaining_size, ask['quantity'])
            weighted_price += executed_qty * ask['price']
            remaining_size -= executed_qty
        
        if order_size - remaining_size <= 0:
            return 0.0
            
        avg_execution_price = weighted_price / (order_size - remaining_size)
        market_impact = (avg_execution_price - base_price) / base_price
        
        return market_impact
    
def prepare_enhanced_features(df: pd.DataFrame, commission: float, 
                             future_periods: int = 5, 
                             feature_engineering: bool = True,
                             remove_outliers: bool = True,
                             add_regime_features: bool = True,
                             for_prediction: bool = False,
                             use_cache: bool = True) -> pd.DataFrame:
    """
    Расширенная подготовка признаков для обучения модели с кэшированием.
    """
    try:
        if df.empty or len(df) < 50:
            logger.warning("Недостаточно данных для подготовки признаков")
            return pd.DataFrame()
        
        # Проверяем кэш только для предсказаний
        if for_prediction and use_cache:
            cached = _indicator_cache.get(df)
            if cached is not None:
                logger.debug("Использован кэш индикаторов")
                return cached
        
        # Копируем DataFrame
        df_feat = df.copy()
        
        # Рассчитываем технические индикаторы
        df_feat = calculate_enhanced_indicators(df_feat)
        
        if df_feat.empty:
            logger.warning("Не удалось рассчитать индикаторы")
            return pd.DataFrame()
        
        # ВАЖНО: Удаляем все признаки с информацией из будущего
        future_columns = ['future_return', 'future_return_1', 'future_return_5', 'future_return_10',
                         'max_future_return', 'min_future_return', 'target_regression']
        
        if for_prediction:
            # При предсказании удаляем ВСЕ признаки из будущего
            df_feat = df_feat.drop(columns=[col for col in future_columns if col in df_feat.columns])
            
            # Сохраняем в кэш
            if use_cache:
                _indicator_cache.set(df, df_feat)
        else:
            # При обучении создаем целевую переменную
            # Рассчитываем будущую доходность
            df_feat['future_return'] = df_feat['close'].pct_change(future_periods).shift(-future_periods)
            
            # Создаем бинарную целевую переменную с учетом комиссии
            threshold = commission * 2  # Порог должен покрывать комиссию на вход и выход
            df_feat['target'] = (df_feat['future_return'] > threshold).astype(int)
            
            # Удаляем строки с NaN в целевой переменной
            df_feat = df_feat.dropna(subset=['target'])
            
            # ВАЖНО: Удаляем признаки из будущего из обучающих данных
            columns_to_drop = [col for col in future_columns if col in df_feat.columns and col != 'target']
            df_feat = df_feat.drop(columns=columns_to_drop)
        
        # Дополнительная инженерия признаков (только если не используем кэш)
        if feature_engineering and (not for_prediction or not use_cache):
            # Взаимодействия между признаками
            if 'rsi' in df_feat.columns and 'macd' in df_feat.columns:
                df_feat['rsi_macd'] = df_feat['rsi'] * df_feat['macd']
            
            if 'rsi' in df_feat.columns and 'bb_position' in df_feat.columns:
                df_feat['bb_rsi'] = df_feat['bb_position'] * df_feat['rsi']
            
            df_feat['volume_price_ratio'] = df_feat['volume'] / (df_feat['close'] + 1e-10)
            
            # Признаки на основе свечных паттернов
            df_feat['body_size'] = np.abs(df_feat['close'] - df_feat['open']) / (df_feat['open'] + 1e-10)
            df_feat['upper_shadow'] = (df_feat['high'] - np.maximum(df_feat['open'], df_feat['close'])) / (df_feat['open'] + 1e-10)
            df_feat['lower_shadow'] = (np.minimum(df_feat['open'], df_feat['close']) - df_feat['low']) / (df_feat['open'] + 1e-10)
            df_feat['is_bullish'] = (df_feat['close'] > df_feat['open']).astype(int)
            
            # Признаки на основе объема
            df_feat['volume_delta'] = df_feat['volume'].diff()
            df_feat['volume_delta_ratio'] = df_feat['volume_delta'] / (df_feat['volume'].shift(1) + 1e-10)
        
        # Добавление признаков рыночного режима
        if add_regime_features and 'ma_50' in df_feat.columns:
            # Определение тренда
            df_feat['trend_up'] = (df_feat['close'] > df_feat['ma_50']).astype(int)
            df_feat['trend_down'] = (df_feat['close'] < df_feat['ma_50']).astype(int)
            
            # Определение волатильности
            if 'volatility' in df_feat.columns:
                volatility_percentile = df_feat['volatility'].rolling(100).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
                )
                df_feat['high_volatility'] = (volatility_percentile > 0.7).astype(int)
                df_feat['low_volatility'] = (volatility_percentile < 0.3).astype(int)
        
        # Удаление выбросов
        if remove_outliers and not for_prediction:
            # Определяем числовые колонки
            numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
            
            # Исключаем целевые переменные
            exclude_cols = ['target', 'hour', 'day_of_week']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Удаляем выбросы
            for col in numeric_cols:
                if col in df_feat.columns:
                    q1 = df_feat[col].quantile(0.01)
                    q99 = df_feat[col].quantile(0.99)
                    df_feat[col] = df_feat[col].clip(q1, q99)
        
        # Заменяем бесконечные значения
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        
        # Обработка NaN
        if for_prediction:
            # Для предсказания заполняем NaN
            df_feat = df_feat.fillna(method='ffill').fillna(0)
        else:
            # Для обучения удаляем строки с большим количеством NaN
            nan_threshold = len(df_feat.columns) * 0.3
            df_feat = df_feat.dropna(thresh=len(df_feat.columns) - nan_threshold)
            # Заполняем оставшиеся NaN
            df_feat = df_feat.fillna(method='ffill').fillna(0)
        
        # Удаляем колонки с постоянными значениями
        if len(df_feat) > 0:
            constant_columns = [col for col in df_feat.columns if df_feat[col].nunique() <= 1]
            if constant_columns:
                logger.debug(f"Удаляем константные колонки: {constant_columns}")
                df_feat = df_feat.drop(columns=constant_columns)
        
        # Финальная проверка
        if df_feat.empty:
            logger.warning("DataFrame пустой после обработки")
            return pd.DataFrame()
        
        # Логируем только каждую 10-ю операцию для ускорения
        if not hasattr(prepare_enhanced_features, '_log_counter'):
            prepare_enhanced_features._log_counter = 0
        
        prepare_enhanced_features._log_counter += 1
        if prepare_enhanced_features._log_counter % 10 == 0:
            logger.info(f"Подготовлено {len(df_feat)} строк с {len(df_feat.columns)} признаками")
        
        if not for_prediction and 'target' in df_feat.columns:
            class_distribution = df_feat['target'].value_counts(normalize=True)
            if prepare_enhanced_features._log_counter % 10 == 0:
                logger.info(f"Распределение целевой переменной: {class_distribution}")
            
            # Проверка баланса классов
            if class_distribution.min() < 0.1:
                logger.warning("Сильный дисбаланс классов!")
        
        return df_feat
        
    except Exception as e:
        import traceback
        logger.error(f"Ошибка подготовки признаков: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()
    
def validate_training_data(df: pd.DataFrame, target_column: str = 'target') -> Tuple[bool, str]:
    """
    Валидация данных перед обучением модели
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Проверка на пустоту
        if df.empty:
            return False, "DataFrame пустой"
        
        # Проверка наличия целевой переменной
        if target_column not in df.columns:
            return False, f"Отсутствует целевая переменная '{target_column}'"
        
        # Проверка на достаточность данных
        if len(df) < 200:
            return False, f"Недостаточно данных: {len(df)} строк (минимум 200)"
        
        # Проверка баланса классов
        class_counts = df[target_column].value_counts()
        if len(class_counts) < 2:
            return False, f"Недостаточно классов в целевой переменной: {class_counts}"
        
        min_class_ratio = class_counts.min() / class_counts.sum()
        if min_class_ratio < 0.05:
            return False, f"Слишком сильный дисбаланс классов: {min_class_ratio:.2%}"
        
        # Проверка на константные признаки
        feature_columns = [col for col in df.columns if col != target_column]
        constant_features = []
        for col in feature_columns:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if len(constant_features) > len(feature_columns) * 0.5:
            return False, f"Слишком много константных признаков: {len(constant_features)}"
        
        # Проверка на NaN
        nan_ratio = df.isna().sum().sum() / (len(df) * len(df.columns))
        if nan_ratio > 0.3:
            return False, f"Слишком много пропущенных значений: {nan_ratio:.2%}"
        
        # Проверка на бесконечные значения
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            return False, f"Обнаружены бесконечные значения: {inf_count}"
        
        return True, "Валидация пройдена успешно"
        
    except Exception as e:
        return False, f"Ошибка валидации: {str(e)}"

