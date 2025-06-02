import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import logging
from typing import Tuple, List, Dict, Optional
from collections import deque
import datetime
import traceback

logger = logging.getLogger("ml_models")

# --- Ансамблевая модель ---
class EnsembleModel:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        self.weights = {'rf': 0.4, 'xgb': 0.6}  # Перераспределили веса после удаления LightGBM
        self.feature_importance = {}
        
    def fit(self, X, y):
        """Обучение всех моделей в ансамбле"""
        for name, model in self.models.items():
            logger.info(f"Обучение модели {name}")
            model.fit(X, y)
            
            # Сохранение важности признаков
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
                
    def predict_proba(self, X):
        """Взвешенное предсказание ансамбля"""
        predictions = np.zeros((X.shape[0], 2))
        
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            predictions += pred * self.weights[name]
            
        return predictions
    
    def get_feature_importance(self, feature_names):
        """Получение усредненной важности признаков"""
        if not self.feature_importance:
            return None
            
        avg_importance = np.zeros(len(feature_names))
        
        for name, importance in self.feature_importance.items():
            avg_importance += importance * self.weights[name]
            
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

# --- Адаптивный менеджер порогов ---
class AdaptiveThresholdManager:
    def __init__(self, window: int = 100):
        self.window = window
        self.threshold_history = deque(maxlen=window)
        self.performance_history = deque(maxlen=window)
        
    def calculate_threshold(self, df: pd.DataFrame, market_conditions, 
                          base_threshold: float = 0.52) -> float:
        """Расчет адаптивного порога на основе рыночных условий и истории"""
        # Базовая корректировка по волатильности
        volatility_adjustment = market_conditions.volatility_percentile / 2000
        
        # Корректировка по режиму рынка
        regime_adjustments = {
            "high_volatility": 0.02,
            "trending_up": -0.02,
            "trending_down": -0.02,
            "ranging": 0.01,
            "unknown": 0.00
        }
        regime_adjustment = regime_adjustments.get(market_conditions.regime, 0)
        
        # Корректировка по производительности
        performance_adjustment = 0
        if len(self.performance_history) >= 20:
            recent_accuracy = np.mean(list(self.performance_history)[-20:])
            if recent_accuracy < 0.45:
                performance_adjustment = 0.01
            elif recent_accuracy > 0.65:
                performance_adjustment = -0.01
                
        # Временная корректировка
        current_hour = datetime.datetime.now().hour
        time_adjustment = 0
        if current_hour in [14, 15, 16]:  # Открытие US рынка
            time_adjustment = 0.005
        elif current_hour in [2, 3, 4]:  # Низкая ликвидность
            time_adjustment = 0.01
            
        # Итоговый порог
        final_threshold = base_threshold + volatility_adjustment + regime_adjustment + \
                        performance_adjustment + time_adjustment
                        
        return np.clip(final_threshold, 0.50, 0.65)
    
    def update_performance(self, predicted: bool, actual: bool):
        """Обновление истории производительности"""
        self.performance_history.append(predicted == actual)

# --- Улучшенный анализатор сигналов ---
class EnhancedSignalAnalyzer:
    def __init__(self, commission: float, spread: float):
        self.commission = commission
        self.spread = spread
        self.signal_history = deque(maxlen=100)
        
    def analyze_signal(self, df: pd.DataFrame, prob_up: float, market_conditions,
                      confidence_score: float) -> Tuple[str, float, str, float]:
        """Комплексный анализ торговых сигналов с учетом рыночных условий"""
        try:
            if df.empty:
                logger.warning("Получен пустой DataFrame для анализа сигналов")
                return "HOLD", 0.5, "Недостаточно данных", 0.0
                
            last = df.iloc[-1]
            signals = []
            signal_strengths = []
            
            # 1. Трендовые сигналы
            trend_signal, trend_strength = self._analyze_trend(df, last)
            signals.append(trend_signal)
            signal_strengths.append(trend_strength * 0.8)
            
            # 2. Моментум сигналы
            momentum_signal, momentum_strength = self._analyze_momentum(df, last)
            signals.append(momentum_signal)
            signal_strengths.append(momentum_strength)
            
            # 3. Объемные сигналы
            volume_signal, volume_strength = self._analyze_volume(df, last)
            signals.append(volume_signal)
            signal_strengths.append(volume_strength * 0.7)
            
            # 4. Паттерны
            pattern_signal, pattern_strength = self._analyze_patterns(df, last)
            signals.append(pattern_signal)
            signal_strengths.append(pattern_strength * 0.6)
            
            # 5. ML сигнал - увеличиваем вес
            ml_signal = 1 if prob_up > 0.5 else -1
            ml_strength = abs(prob_up - 0.5) * 3
            signals.append(ml_signal)
            signal_strengths.append(ml_strength)
            
            # Взвешенное голосование
            weighted_signal = np.average(signals, weights=signal_strengths)
            overall_strength = np.mean(signal_strengths)
            
            # Корректировка по рыночным условиям
            if market_conditions.regime == "high_volatility":
                threshold_multiplier = 1.2
            elif market_conditions.regime in ["trending_up", "trending_down"]:
                threshold_multiplier = 0.7
            else:
                threshold_multiplier = 1.0
                
            # Формирование финального сигнала
            signal_threshold = 0.2 * threshold_multiplier
            
            if weighted_signal > signal_threshold and overall_strength > 0.3:
                signal = "BUY"
                reason = self._generate_reason(signals, signal_strengths, "BUY", market_conditions)
            elif weighted_signal < -signal_threshold and overall_strength > 0.3:
                signal = "SELL"
                reason = self._generate_reason(signals, signal_strengths, "SELL", market_conditions)
            else:
                signal = "HOLD"
                reason = f"Слабый сигнал: {weighted_signal:.2f}, сила: {overall_strength:.2f}"
                
            # Расчет уверенности
            final_confidence = confidence_score * overall_strength
            
            # Сохранение в историю
            self.signal_history.append({
                'timestamp': datetime.datetime.now(),
                'signal': signal,
                'weighted_signal': weighted_signal,
                'strength': overall_strength,
                'confidence': final_confidence,
                'market_regime': market_conditions.regime
            })
            
            return signal, overall_strength, reason, final_confidence
            
        except Exception as e:
            logger.error(f"Ошибка анализа сигналов: {e}")
            logger.error(traceback.format_exc())
            return "HOLD", 0.0, f"Ошибка анализа: {e}", 0.0
    
    def _analyze_trend(self, df: pd.DataFrame, last: pd.Series) -> Tuple[int, float]:
        """Анализ трендовых индикаторов"""
        signal = 0
        strength = 0
        
        # EMA кроссовер
        if 'ma_20' in last and 'ma_50' in last:
            if last['ma_20'] > last['ma_50']:
                signal += 1
                strength += abs(last['ma_20'] - last['ma_50']) / last['close']
            else:
                signal -= 1
                strength += abs(last['ma_20'] - last['ma_50']) / last['close']
                
        # ADX для силы тренда
        if 'adx' in df.columns and last.get('adx', 0) > 25:
            strength *= 1.5
            
        # MACD
        if 'macd_hist' in last:
            if last['macd_hist'] > 0:
                signal += 1
            else:
                signal -= 1
            strength += abs(last['macd_hist']) / last['close'] * 100
            
        return np.sign(signal), min(strength, 1.0)
    
    def _analyze_momentum(self, df: pd.DataFrame, last: pd.Series) -> Tuple[int, float]:
        """Анализ моментум индикаторов"""
        signal = 0
        strength = 0
        
        # RSI
        if 'rsi' in last:
            if last['rsi'] < 30:
                signal += 2
                strength += (30 - last['rsi']) / 30
            elif last['rsi'] > 70:
                signal -= 2
                strength += (last['rsi'] - 70) / 30
            else:
                strength += 0.3
                
        # Стохастик
        if 'stoch_k' in last and 'stoch_d' in last:
            if last['stoch_k'] < 20 and last['stoch_k'] > last['stoch_d']:
                signal += 1
                strength += 0.3
            elif last['stoch_k'] > 80 and last['stoch_k'] < last['stoch_d']:
                signal -= 1
                strength += 0.3
                
        # MFI
        if 'mfi' in last:
            if last['mfi'] < 20:
                signal += 1
                strength += 0.2
            elif last['mfi'] > 80:
                signal -= 1
                strength += 0.2
                
        return np.sign(signal), min(strength, 1.0)
    
    def _analyze_volume(self, df: pd.DataFrame, last: pd.Series) -> Tuple[int, float]:
        """Анализ объемных индикаторов"""
        signal = 0
        strength = 0
        
        # Volume ratio
        if 'volume_ratio' in last:
            if last['volume_ratio'] > 1.5:
                strength += 0.5
                # Определяем направление по цене
                if 'return_1m' in last and last['return_1m'] > 0:
                    signal += 1
                else:
                    signal -= 1
                    
        # OBV slope
        if 'obv_slope' in last:
            if last['obv_slope'] > 0:
                signal += 1
                strength += min(abs(last['obv_slope']) * 10, 0.5)
            else:
                signal -= 1
                strength += min(abs(last['obv_slope']) * 10, 0.5)
                
        return np.sign(signal), min(strength, 1.0)
    
    def _analyze_patterns(self, df: pd.DataFrame, last: pd.Series) -> Tuple[int, float]:
        """Анализ свечных паттернов и уровней"""
        signal = 0
        strength = 0
        
        # Свечные паттерны
        if last.get('hammer', 0) == 1:
            signal += 2
            strength += 0.7
        elif last.get('shooting_star', 0) == 1:
            signal -= 2
            strength += 0.7
            
        # Позиция относительно Bollinger Bands
        if 'bb_position' in last:
            if last['bb_position'] < 0.2:
                signal += 1
                strength += 0.4
            elif last['bb_position'] > 0.8:
                signal -= 1
                strength += 0.4
                
        # VWAP
        if 'price_to_vwap' in last:
            if last['price_to_vwap'] < 0.98:
                signal += 1
                strength += 0.3
            elif last['price_to_vwap'] > 1.02:
                signal -= 1
                strength += 0.3
                
        return np.sign(signal), min(strength, 1.0)
    
    def _generate_reason(self, signals: List[int], strengths: List[float], 
                        final_signal: str, market_conditions) -> str:
        """Генерация описания причины сигнала"""
        components = []
        signal_names = ["Тренд", "Моментум", "Объем", "Паттерны", "ML"]
        
        for i, (sig, strength, name) in enumerate(zip(signals, strengths, signal_names)):
            if strength > 0.3:
                direction = "↑" if sig > 0 else "↓" if sig < 0 else "→"
                components.append(f"{name}{direction}({strength:.1f})")
                
        reason = f"Режим: {market_conditions.regime}, Сигналы: {', '.join(components)}"
        return reason

# --- Функции обучения и предсказания ---
# В функции train_enhanced_model добавьте:

def train_enhanced_model(df: pd.DataFrame, commission: float) -> Tuple[EnsembleModel, StandardScaler, List[str]]:
    """Обучение улучшенной модели"""
    try:
        from calculations import prepare_enhanced_features
        
        # Подготовка признаков
        df_feat = prepare_enhanced_features(df, commission)
        
        if df_feat.empty or len(df_feat) < 200:  # Увеличили минимум данных
            logger.warning("Недостаточно данных после подготовки признаков")
            return None, None, []
        
        # Используем альтернативную целевую переменную для лучшего обучения
        target_column = 'target_multi' if 'target_multi' in df_feat.columns else 'target'
        
        # Список признаков
        feature_columns = [col for col in df_feat.columns if col not in 
                         ['target', 'target_multi', 'future_return', 'max_future_return']]
        
        X = df_feat[feature_columns].values
        y = df_feat[target_column].values
        
        # Проверка баланса классов
        class_counts = np.bincount(y)
        if len(class_counts) < 2 or min(class_counts) < 30:  # Увеличили минимум
            logger.warning(f"Несбалансированные классы: {class_counts}")
            # Применяем SMOTE для балансировки
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        
        # Масштабирование с RobustScaler для устойчивости к выбросам
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Отбор признаков
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=min(50, len(feature_columns)))
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        
        # Временное разделение с большим количеством фолдов
        tscv = TimeSeriesSplit(n_splits=10)
        best_score = 0
        best_model = None
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
            if len(val_idx) < 50:  # Пропускаем слишком маленькие валидационные наборы
                continue
                
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Обучение ансамбля с оптимизированными параметрами
            ensemble = EnsembleModel()
            
            # Настройка гиперпараметров для RandomForest
            ensemble.models['rf'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # Настройка гиперпараметров для XGBoost
            ensemble.models['xgb'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
            )
            
            ensemble.fit(X_train, y_train)
            
            # Оценка
            y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
            
            if len(np.unique(y_val)) > 1:
                score = roc_auc_score(y_val, y_pred_proba)
                scores.append(score)
                logger.info(f"Fold {fold + 1} AUC: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = ensemble
        
        if best_model is None:
            logger.warning("Не удалось найти лучшую модель, обучаем на всех данных")
            best_model = EnsembleModel()
            best_model.fit(X_selected, y)
        
        # Вычисляем среднюю оценку
        avg_score = np.mean(scores) if scores else 0.5
        logger.info(f"Модель обучена. Средний AUC: {avg_score:.4f}, Лучший AUC: {best_score:.4f}")
        
        # Анализ важности признаков
        feature_importance = best_model.get_feature_importance(selected_features)
        if feature_importance is not None:
            logger.info(f"Топ-15 важных признаков:\n{feature_importance.head(15)}")
        
        return best_model, scaler, selected_features
        
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        logger.error(traceback.format_exc())
        return None, None, []

def predict_enhanced_direction(
    df: pd.DataFrame,
    model: EnsembleModel,
    scaler,
    features: List[str],
    commission: float  # оставим для совместимости, вдруг пригодится далее
) -> Tuple[int, float]:
    """
    Предсказание направления движения рынка с использованием ансамблевой модели.
    Возвращает: бинарный лейбл (0/1) и вероятность движения вверх.
    """
    from calculations import calculate_enhanced_indicators

    if df.empty or model is None or scaler is None or not features:
        logger.warning("[predict_enhanced_direction] Пустые входные данные")
        return 0, 0.5

    try:
        df_feat = calculate_enhanced_indicators(df)  # ❗ Убираем commission
        if df_feat.empty or len(df_feat) < 1:
            logger.warning("[predict_enhanced_direction] Пустой DataFrame после обработки индикаторов")
            return 0, 0.5

        latest_row = df_feat.iloc[-1]

        # Проверим, что все признаки присутствуют
        missing_features = [f for f in features if f not in latest_row.index]
        if missing_features:
            logger.warning(f"[predict_enhanced_direction] Отсутствуют признаки: {missing_features}")
            return 0, 0.5

        X = latest_row[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0][1]

        label = int(proba > 0.5)
        return label, proba

    except Exception as e:
        logger.error(f"[predict_enhanced_direction] Ошибка: {e}")
        logger.error(traceback.format_exc())
        return 0, 0.5


# --- Бэктестер ---
class Backtester:
    def __init__(self, initial_balance: float = 10000, commission: float = 0.001):
        self.initial_balance = initial_balance
        self.commission = commission
        self.reset()
        
    def reset(self):
        """Сброс состояния бэктестера"""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.in_position = False
        self.current_position = None
        
    def backtest_strategy(self, df: pd.DataFrame, model, scaler, features, 
                         signal_analyzer: EnhancedSignalAnalyzer,
                         market_detector,
                         threshold_manager: AdaptiveThresholdManager) -> dict:
        """Бэктест стратегии на исторических данных"""
        from calculations import calculate_enhanced_indicators
        
        self.reset()
        
        # Минимальное количество данных для начала
        min_lookback = 200
        
        for i in range(min_lookback, len(df)):
            current_data = df.iloc[:i]
            current_price = df.iloc[i]['close']
            
            # Расчет индикаторов для текущих данных
            current_data = calculate_enhanced_indicators(current_data)
            if current_data.empty:
                continue
            
            # Определение рыночных условий
            market_conditions = market_detector.detect_regime(current_data)
            
            # ML предсказание
            label, prob_up = predict_enhanced_direction(current_data, model, scaler, features, self.commission)
            
            # Расчет адаптивного порога
            threshold = threshold_manager.calculate_threshold(current_data, market_conditions)
            
            # Расчет уверенности
            confidence_score = abs(prob_up - 0.5) * 2
            
            # Анализ сигналов
            signal, strength, reason, confidence = signal_analyzer.analyze_signal(
                current_data, prob_up, market_conditions, confidence_score
            )
            
            # Управление позициями
            if not self.in_position and signal in ["BUY", "SELL"]:
                # Открытие позиции
                position_size = self.balance * 0.02 / current_price  # 2% риска
                
                self.current_position = {
                    'type': signal,
                    'entry_price': current_price,
                    'size': position_size,
                    'entry_time': df.index[i],
                    'market_regime': market_conditions.regime,
                    'confidence': confidence
                }
                self.in_position = True
                
            elif self.in_position:
                # Проверка условий закрытия
                if self.current_position['type'] == "BUY":
                    # Простой стоп-лосс и тейк-профит
                    profit_pct = (current_price - self.current_position['entry_price']) / \
                            self.current_position['entry_price']
                    
                    if profit_pct < -0.02 or profit_pct > 0.04 or signal == "SELL":
                        # Закрытие позиции
                        self._close_position(current_price, df.index[i])
                        
                elif self.current_position['type'] == "SELL":
                    profit_pct = (self.current_position['entry_price'] - current_price) / \
                            self.current_position['entry_price']
                    
                    if profit_pct < -0.02 or profit_pct > 0.04 or signal == "BUY":
                        self._close_position(current_price, df.index[i])
            
            # Обновление кривой эквити
            current_equity = self._calculate_equity(current_price)
            self.equity_curve.append(current_equity)
        
        # Расчет метрик
        return self._calculate_backtest_metrics()
    
    def _close_position(self, exit_price: float, exit_time):
        """Закрытие позиции в бэктесте"""
        if not self.current_position:
            return
            
        # Расчет прибыли
        if self.current_position['type'] == "BUY":
            gross_profit = self.current_position['size'] * (exit_price - self.current_position['entry_price'])
        else:
            gross_profit = self.current_position['size'] * (self.current_position['entry_price'] - exit_price)
            
        # Вычитание комиссии
        commission_cost = self.current_position['size'] * exit_price * self.commission * 2
        net_profit = gross_profit - commission_cost
        
        # Обновление баланса
        self.balance += net_profit
        
        # Запись сделки
        self.trades.append({
            'type': self.current_position['type'],
            'entry_price': self.current_position['entry_price'],
            'exit_price': exit_price,
            'size': self.current_position['size'],
            'profit': net_profit,
            'profit_pct': net_profit / (self.current_position['size'] * self.current_position['entry_price']) * 100,
            'entry_time': self.current_position['entry_time'],
            'exit_time': exit_time,
            'market_regime': self.current_position['market_regime'],
            'confidence': self.current_position['confidence']
        })
        
        self.in_position = False
        self.current_position = None
    
    def _calculate_equity(self, current_price: float) -> float:
        """Расчет текущего эквити"""
        equity = self.balance
        
        if self.in_position and self.current_position:
            # Добавляем нереализованную прибыль
            if self.current_position['type'] == "BUY":
                unrealized = self.current_position['size'] * (current_price - self.current_position['entry_price'])
            else:
                unrealized = self.current_position['size'] * (self.current_position['entry_price'] - current_price)
            
            equity += unrealized
            
        return equity
    
    def _calculate_backtest_metrics(self) -> dict:
        """Расчет метрик бэктеста"""
        if not self.trades:
            return {
                'total_return': 0,
                'final_balance': self.balance,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'regime_analysis': {},
                'equity_curve': self.equity_curve
            }
            
        # Базовые метрики
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        profits = [t['profit'] for t in self.trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # Profit factor
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0.0001
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Максимальная просадка
        equity_array = np.array(self.equity_curve)
        if len(equity_array) > 1:
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (running_max - equity_array) / running_max * 100
            max_drawdown = np.max(drawdown)
        else:
            max_drawdown = 0
        
        # Sharpe ratio
        if len(equity_array) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            if np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Анализ по режимам рынка
        regime_analysis = {}
        if self.trades:
            for regime in set(t.get('market_regime', 'unknown') for t in self.trades):
                regime_trades = [t for t in self.trades if t.get('market_regime', 'unknown') == regime]
                regime_profits = [t['profit'] for t in regime_trades]
                regime_wins = [p for p in regime_profits if p > 0]
                
                regime_analysis[regime] = {
                    'trades': len(regime_trades),
                    'win_rate': len(regime_wins) / len(regime_trades) if regime_trades else 0,
                    'avg_profit': np.mean(regime_profits) if regime_profits else 0
                }
        
        return {
            'total_return': total_return,
            'final_balance': self.balance,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'best_trade': max(profits) if profits else 0,
            'worst_trade': min(profits) if profits else 0,
            'regime_analysis': regime_analysis,
            'equity_curve': self.equity_curve
        }
