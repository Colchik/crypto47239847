import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
import datetime
import talib
from typing import Any, Dict

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
        """Расчет индикатора RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        loss = loss.replace(0, np.finfo(float).eps)
        rs = gain / loss
        return 100 - (100 / (1 + rs))

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
        """Расчет Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()
        
        mfi_ratio = positive_flow_sum / negative_flow_sum.replace(0, np.finfo(float).eps)
        return 100 - (100 / (1 + mfi_ratio))
    
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

# --- Функции расчета индикаторов ---
def calculate_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Расчет всех технических индикаторов"""
    try:
        if df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        ti = TechnicalIndicators()
        
        # Базовые индикаторы
        df['rsi'] = ti.calculate_rsi(df['close'])
        df['rsi_ma'] = df['rsi'].rolling(10).mean()
        
        df['atr'] = ti.calculate_atr(df['high'], df['low'], df['close'])
        df['atr_norm'] = df['atr'] / df['close']
        
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
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
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
        df['price_to_vwap'] = df['close'] / df['vwap']
        
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
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                df[f'ma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        return df.dropna()
        
    except Exception as e:
        logger.error(f"Ошибка расчета индикаторов: {e}")
        return pd.DataFrame()

# --- Подготовка признаков для ML ---
def prepare_enhanced_features(df: pd.DataFrame, commission: float) -> pd.DataFrame:
    """Расширенная подготовка признаков для модели"""
    try:
        df_feat = df.copy()
        
        if df_feat.empty:
            logger.warning("Получен пустой DataFrame для подготовки признаков")
            return pd.DataFrame()
        
        # Убедимся, что все индикаторы рассчитаны
        if 'rsi' not in df_feat.columns:
            df_feat = calculate_enhanced_indicators(df_feat)
        
        # Ценовые паттерны
        df_feat['higher_high'] = (df_feat['high'] > df_feat['high'].shift(1)).astype(int)
        df_feat['lower_low'] = (df_feat['low'] < df_feat['low'].shift(1)).astype(int)
        df_feat['inside_bar'] = ((df_feat['high'] < df_feat['high'].shift(1)) & 
                                 (df_feat['low'] > df_feat['low'].shift(1))).astype(int)
        
        # Лаговые признаки
        for lag in range(1, 10):
            df_feat[f'return_lag_{lag}'] = df_feat['close'].pct_change(lag)
            df_feat[f'volume_lag_{lag}'] = df_feat['volume'].shift(lag)
            
        # Возвраты различных периодов
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 60]:
            df_feat[f'return_{period}m'] = df_feat['close'].pct_change(period)
            df_feat[f'volume_change_{period}m'] = df_feat['volume'].pct_change(period)
            
        # Скользящие средние и их производные
        for period in [5, 10, 20, 50]:
            if f'ma_{period}' in df_feat.columns:
                df_feat[f'ma_{period}_slope'] = df_feat[f'ma_{period}'].pct_change(5)
                df_feat[f'price_to_ma_{period}'] = df_feat['close'] / df_feat[f'ma_{period}']
                
            if f'ema_{period}' in df_feat.columns:
                df_feat[f'ema_{period}_slope'] = df_feat[f'ema_{period}'].pct_change(5)
                df_feat[f'price_to_ema_{period}'] = df_feat['close'] / df_feat[f'ema_{period}']
        
        # Кроссоверы скользящих средних
        if 'ma_20' in df_feat.columns and 'ma_50' in df_feat.columns:
            df_feat['ma_cross_20_50'] = (df_feat['ma_20'] > df_feat['ma_50']).astype(int)
            df_feat['ma_cross_20_50_distance'] = (df_feat['ma_20'] - df_feat['ma_50']) / df_feat['close']
            
        # Микроструктура рынка
        df_feat['bid_ask_spread'] = (df_feat['high'] - df_feat['low']) / df_feat['close']
        df_feat['price_position'] = (df_feat['close'] - df_feat['low']) / (df_feat['high'] - df_feat['low'] + 1e-10)
        df_feat['high_low_ratio'] = df_feat['high'] / df_feat['low']
        df_feat['close_to_high'] = (df_feat['high'] - df_feat['close']) / df_feat['high']
        df_feat['close_to_low'] = (df_feat['close'] - df_feat['low']) / df_feat['low']
        
        # Паттерны свечей
        body = abs(df_feat['close'] - df_feat['open'])
        full_range = df_feat['high'] - df_feat['low']
        upper_shadow = df_feat['high'] - df_feat[['close', 'open']].max(axis=1)
        lower_shadow = df_feat[['close', 'open']].min(axis=1) - df_feat['low']
        
        df_feat['body_ratio'] = body / (full_range + 1e-10)
        df_feat['upper_shadow_ratio'] = upper_shadow / (full_range + 1e-10)
        df_feat['lower_shadow_ratio'] = lower_shadow / (full_range + 1e-10)
        
        df_feat['doji'] = (body / (full_range + 1e-10) < 0.1).astype(int)
        df_feat['hammer'] = ((lower_shadow > 2 * body) & 
                            (upper_shadow < body * 0.3) & 
                            (df_feat['close'] > df_feat['open'])).astype(int)
        df_feat['shooting_star'] = ((upper_shadow > 2 * body) & 
                                   (lower_shadow < body * 0.3) & 
                                   (df_feat['close'] < df_feat['open'])).astype(int)
        df_feat['bullish_engulfing'] = ((df_feat['close'] > df_feat['open']) & 
                                        (df_feat['open'] < df_feat['close'].shift(1)) & 
                                        (df_feat['close'] > df_feat['open'].shift(1))).astype(int)
        df_feat['bearish_engulfing'] = ((df_feat['close'] < df_feat['open']) & 
                                        (df_feat['open'] > df_feat['close'].shift(1)) & 
                                        (df_feat['close'] < df_feat['open'].shift(1))).astype(int)
        
        # Временные признаки
        if isinstance(df_feat.index, pd.DatetimeIndex):
            df_feat['hour'] = df_feat.index.hour
            df_feat['day_of_week'] = df_feat.index.dayofweek
            df_feat['day_of_month'] = df_feat.index.day
            df_feat['is_asian_session'] = df_feat['hour'].between(0, 8).astype(int)
            df_feat['is_european_session'] = df_feat['hour'].between(8, 16).astype(int)
            df_feat['is_us_session'] = df_feat['hour'].between(16, 24).astype(int)
            df_feat['is_weekend'] = df_feat['day_of_week'].isin([5, 6]).astype(int)
            
            # Циклические временные признаки
            df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
            df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
            df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
            df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
            
        # Признаки тренда
        if 'ma_20' in df_feat.columns and 'ma_50' in df_feat.columns:
            df_feat['trend_strength'] = abs(df_feat['ma_20'] - df_feat['ma_50']) / df_feat['close']
            df_feat['trend_consistency'] = (df_feat['close'] > df_feat['ma_20']).rolling(10).mean()
        
        # Признаки импульса
        for period in [5, 10, 20, 50]:
            df_feat[f'momentum_{period}'] = df_feat['close'] / df_feat['close'].shift(period) - 1
            
        # Статистические признаки
        for period in [10, 20, 50]:
            if len(df_feat) >= period:
                df_feat[f'return_mean_{period}'] = df_feat['close'].pct_change().rolling(period).mean()
                df_feat[f'return_std_{period}'] = df_feat['close'].pct_change().rolling(period).std()
                df_feat[f'return_skew_{period}'] = df_feat['close'].pct_change().rolling(period).skew()
                df_feat[f'return_kurt_{period}'] = df_feat['close'].pct_change().rolling(period).kurt()
                
                # Z-score
                df_feat[f'zscore_{period}'] = (df_feat['close'] - df_feat['close'].rolling(period).mean()) / df_feat['close'].rolling(period).std()
        
        # Признаки основанные на объеме
        df_feat['volume_price_trend'] = (df_feat['volume'] * df_feat['close'].pct_change()).cumsum()
        df_feat['accumulation_distribution'] = ((df_feat['close'] - df_feat['low']) - (df_feat['high'] - df_feat['close'])) / (df_feat['high'] - df_feat['low']) * df_feat['volume']
        df_feat['accumulation_distribution'] = df_feat['accumulation_distribution'].cumsum()
        
        # Признаки волатильности
        for period in [5, 10, 20]:
            if len(df_feat) >= period:
                df_feat[f'volatility_{period}'] = df_feat['close'].rolling(period).std()
                df_feat[f'volatility_ratio_{period}'] = df_feat[f'volatility_{period}'] / df_feat['close']
                df_feat[f'volatility_change_{period}'] = df_feat[f'volatility_{period}'].pct_change()
        
        # Признаки основанные на индикаторах
        if 'rsi' in df_feat.columns:
            df_feat['rsi_overbought'] = (df_feat['rsi'] > 70).astype(int)
            df_feat['rsi_oversold'] = (df_feat['rsi'] < 30).astype(int)
            df_feat['rsi_divergence'] = df_feat['rsi'].diff() * df_feat['close'].pct_change()
            
        if 'macd_hist' in df_feat.columns:
            df_feat['macd_cross'] = ((df_feat['macd'] > df_feat['macd_signal']) & 
                                     (df_feat['macd'].shift(1) <= df_feat['macd_signal'].shift(1))).astype(int)
            df_feat['macd_divergence'] = df_feat['macd_hist'].diff() * df_feat['close'].pct_change()
            
        if 'stoch_k' in df_feat.columns and 'stoch_d' in df_feat.columns:
            df_feat['stoch_overbought'] = ((df_feat['stoch_k'] > 80) & (df_feat['stoch_d'] > 80)).astype(int)
            df_feat['stoch_oversold'] = ((df_feat['stoch_k'] < 20) & (df_feat['stoch_d'] < 20)).astype(int)
            df_feat['stoch_cross'] = ((df_feat['stoch_k'] > df_feat['stoch_d']) & 
                                      (df_feat['stoch_k'].shift(1) <= df_feat['stoch_d'].shift(1))).astype(int)
        
        # Комбинированные признаки
        if 'rsi' in df_feat.columns and 'mfi' in df_feat.columns:
            df_feat['rsi_mfi_divergence'] = abs(df_feat['rsi'] - df_feat['mfi'])
            
        if 'adx' in df_feat.columns:
            df_feat['strong_trend'] = (df_feat['adx'] > 25).astype(int)
            df_feat['weak_trend'] = (df_feat['adx'] < 20).astype(int)
            
        # Признаки основанные на уровнях
        recent_high = df_feat['high'].rolling(20).max()
        recent_low = df_feat['low'].rolling(20).min()
        df_feat['distance_from_high'] = (recent_high - df_feat['close']) / df_feat['close']
        df_feat['distance_from_low'] = (df_feat['close'] - recent_low) / df_feat['close']
        df_feat['range_position'] = (df_feat['close'] - recent_low) / (recent_high - recent_low + 1e-10)
        
        # Фрактальные признаки
        df_feat['fractal_dimension'] = df_feat['close'].rolling(20).apply(
            lambda x: np.log(np.std(np.diff(x))) / np.log(len(x)) if len(x) > 1 else 0
        )
        
        # Целевая переменная с учетом комиссии и проскальзывания
        df_feat['future_return'] = df_feat['close'].shift(-1) / df_feat['close'] - 1
        
        # Более сложная целевая переменная - учитываем не только направление, но и величину движения
        # Покупаем только если ожидаемая прибыль превышает комиссию в 3 раза
        df_feat['target'] = (df_feat['future_return'] > commission * 3).astype(int)
        
        # Альтернативная целевая переменная - максимальная прибыль в следующие N баров
        future_periods = 5
        future_returns = pd.DataFrame()
        for i in range(1, future_periods + 1):
            future_returns[f'return_{i}'] = df_feat['close'].shift(-i) / df_feat['close'] - 1
        
        df_feat['max_future_return'] = future_returns.max(axis=1)
        df_feat['target_multi'] = (df_feat['max_future_return'] > commission * 3).astype(int)
        
        # Удаление строк с пропущенными значениями
        df_feat.dropna(inplace=True)
        
        # Удаление избыточных колонок
        columns_to_drop = ['open', 'high', 'low', 'close', 'volume']
        df_feat = df_feat.drop(columns=[col for col in columns_to_drop if col in df_feat.columns])
        
        return df_feat
        
    except Exception as e:
        logger.error(f"Ошибка подготовки признаков: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

# --- Анализатор корреляций ---
class CorrelationAnalyzer:
    def __init__(self, symbols: List[str], correlation_window: int = 100):
        self.symbols = symbols
        self.correlation_window = correlation_window
        self.correlation_cache = pd.DataFrame()
        self.last_update = None
        
    def update_correlations(self, client) -> pd.DataFrame:
        """Обновление корреляционной матрицы"""
        try:
            logger.debug(f"Начало обновления корреляций для {len(self.symbols)} символов")
            prices = {}
            
            for symbol in self.symbols:
                try:
                    logger.debug(f"Получение данных для корреляций: {symbol}")
                    from binance_utils import fetch_binance_klines
                    df = fetch_binance_klines(client, symbol, "15m", 1000)
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


