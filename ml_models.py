import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import logging
from typing import Tuple, List, Dict, Optional
from collections import deque
import datetime
import traceback
import matplotlib.pyplot as plt
from boruta import BorutaPy
import shap

logger = logging.getLogger("ml_models")

# --- Ансамблевая модель ---

class EnsembleModel:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.01,
                subsample=0.6,
                colsample_bytree=0.6,
                gamma=0.2,
                reg_alpha=0.2,
                reg_lambda=2,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        self.weights = {'rf': 0.4, 'xgb': 0.6}
        self.feature_importance = {}
        self.is_fitted = False

    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Пустые данные для обучения")
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(f"Недостаточно классов для обучения: {unique_classes}")
        for name, model in self.models.items():
            logger.info(f"Обучение модели {name}")
            model.fit(X, y)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
        self.is_fitted = True

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() перед predict_proba()")
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        predictions = np.zeros((X.shape[0], 2))
        for name, model in self.models.items():
            try:
                pred = model.predict_proba(X)
                if pred.shape[1] != 2:
                    logger.warning(f"Модель {name} вернула {pred.shape[1]} классов вместо 2")
                    binary_pred = np.zeros((pred.shape[0], 2))
                    binary_pred[:, 0] = pred[:, 0]
                    binary_pred[:, 1] = pred[:, -1]
                    row_sums = binary_pred.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1
                    pred = binary_pred / row_sums
                if np.isnan(pred).any():
                    nan_rows = np.isnan(pred).any(axis=1)
                    pred[nan_rows] = [0.5, 0.5]
                predictions += pred * self.weights[name]
            except Exception as e:
                logger.error(f"Ошибка предсказания для модели {name}: {e}")
                predictions += np.full((X.shape[0], 2), 0.5) * self.weights[name]
        if np.isnan(predictions).any():
            predictions = np.nan_to_num(predictions, nan=0.5)
        row_sums = predictions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        predictions /= row_sums
        return predictions

    def get_feature_importance(self, feature_names):
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


# --- BorutaPy отбор признаков ---

def feature_selection_boruta(X, y, feature_columns):
    if len(X) < 100:
        logger.warning(f"Недостаточно данных для BorutaPy: {len(X)} образцов")
        return feature_columns
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logger.warning(f"Недостаточно классов для BorutaPy: {unique_classes}")
        return feature_columns
    try:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        min_class_samples = min(np.sum(y == c) for c in unique_classes)
        if min_class_samples < 10:
            logger.warning(f"Слишком мало образцов в классе: {min_class_samples}")
            return feature_columns[:len(feature_columns)//2]
        selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=50)
        selector.fit(X, y)
        selected_features = [feature_columns[i] for i, x in enumerate(selector.support_) if x]
        if len(selected_features) < 5:
            rf.fit(X, y)
            importances = rf.feature_importances_
            important_indices = np.argsort(importances)[-10:]
            selected_features = [feature_columns[i] for i in important_indices]
        logger.info(f"BorutaPy выбрал {len(selected_features)} признаков из {len(feature_columns)}")
        return selected_features
    except Exception as e:
        logger.error(f"Ошибка в BorutaPy: {e}")
        try:
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            important_indices = np.argsort(importances)[-min(20, len(feature_columns)):]
            return [feature_columns[i] for i in important_indices]
        except:
            return feature_columns[:min(20, len(feature_columns))]


# --- Калибровка вероятностей ---

def calibrate_model(model, X, y):
    if len(X) < 50:
        logger.warning("Недостаточно данных для калибровки, возвращаем исходную модель")
        return model
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logger.warning("Недостаточно классов для калибровки")
        return model
    try:
        n_folds = min(3, len(X) // 50)
        if n_folds < 2:
            logger.warning("Недостаточно данных для кросс-валидации в калибровке")
            return model
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=n_folds)
        calibrated.fit(X, y)
        return calibrated
    except Exception as e:
        logger.warning(f"Ошибка калибровки: {e}, возвращаем исходную модель")
        return model


# --- Автоматический подбор порога ---

def find_best_threshold(y_true, y_proba):
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.4, 0.7, 0.01):
        preds = (y_proba > thr).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


# --- SHAP-анализ ---

def shap_feature_importance(model, X, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        logger.info("SHAP-анализ выполнен успешно.")
    except Exception as e:
        logger.warning(f"SHAP-анализ не выполнен: {e}")

def train_enhanced_model(df: pd.DataFrame, commission: float) -> Tuple[EnsembleModel, RobustScaler, List[str], float, List[str]]:
    try:
        from calculations import prepare_enhanced_features, validate_training_data

        if df.empty or len(df) < 100:  # Уменьшено с 200
            logger.warning(f"Недостаточно данных для обучения: {len(df)} строк")
            return None, None, [], 0.5, []

        df_feat = prepare_enhanced_features(df, commission)
        if df_feat.empty or len(df_feat) < 100:  # Уменьшено с 200
            logger.warning(f"Недостаточно данных после подготовки признаков: {len(df_feat)} строк")
            return None, None, [], 0.5, []

        is_valid, error_message = validate_training_data(df_feat, target_column='target')
        if not is_valid:
            logger.error(f"Данные не прошли валидацию: {error_message}")
            return None, None, [], 0.5, []

        target_column = 'target'
        if target_column not in df_feat.columns:
            logger.error(f"Отсутствует целевая переменная '{target_column}'")
            return None, None, [], 0.5, []

        feature_columns = [col for col in df_feat.columns if col not in ['target', 'target_multi', 'future_return', 'max_future_return']]
        all_feature_columns = feature_columns.copy()

        if not feature_columns:
            logger.error("Нет признаков для обучения")
            return None, None, [], 0.5, []

        # Проверяем типы данных перед извлечением значений
        df_feat_numeric = df_feat[feature_columns].copy()
        
        # Преобразуем все колонки к числовому типу
        for col in df_feat_numeric.columns:
            if df_feat_numeric[col].dtype == 'object' or df_feat_numeric[col].dtype.name == 'category':
                logger.warning(f"Колонка {col} имеет нечисловой тип {df_feat_numeric[col].dtype}, пытаемся преобразовать")
                try:
                    df_feat_numeric[col] = pd.to_numeric(df_feat_numeric[col], errors='coerce')
                except Exception as e:
                    logger.error(f"Не удалось преобразовать колонку {col}: {e}")
                    # Удаляем проблемную колонку
                    df_feat_numeric = df_feat_numeric.drop(columns=[col])
                    feature_columns.remove(col)
        
        # Проверяем на NaN с помощью pandas
        if df_feat_numeric.isnull().any().any():
            logger.warning("Обнаружены NaN значения в данных, заполняем")
            df_feat_numeric = df_feat_numeric.fillna(0)
        
        # Проверяем на бесконечные значения
        inf_mask = np.isinf(df_feat_numeric.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            logger.warning("Обнаружены бесконечные значения, заменяем")
            df_feat_numeric = df_feat_numeric.replace([np.inf, -np.inf], 0)
        
        X = df_feat_numeric.values
        y = df_feat[target_column].values

        # Дополнительная проверка после преобразования в numpy array
        if not np.issubdtype(X.dtype, np.number):
            logger.error(f"X имеет нечисловой тип данных: {X.dtype}")
            return None, None, [], 0.5, []

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.error(f"Недостаточно классов в целевой переменной: {unique_classes}")
            return None, None, [], 0.5, []

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            selected_features = feature_selection_boruta(X_scaled, y, feature_columns)
            if not selected_features:
                logger.warning("BorutaPy не выбрал ни одного признака, используем все")
                selected_features = feature_columns
        except Exception as e:
            logger.warning(f"Ошибка в BorutaPy: {e}, используем все признаки")
            selected_features = feature_columns

        # Убеждаемся, что selected_features содержит только существующие колонки
        selected_features = [f for f in selected_features if f in feature_columns]
        
        X_selected = pd.DataFrame(X_scaled, columns=feature_columns)[selected_features].values

        # Финальная проверка на NaN
        if np.any(np.isnan(X_selected)):
            logger.warning("Обнаружены NaN значения после отбора признаков, заменяем на 0")
            X_selected = np.nan_to_num(X_selected, nan=0.0)

        # Адаптивная настройка параметров в зависимости от количества данных
        n_samples = len(X_selected)
        logger.info(f"Количество образцов для обучения: {n_samples}")

        # Адаптивное количество фолдов
        if n_samples < 200:
            n_splits = 2
        elif n_samples < 500:
            n_splits = 3
        elif n_samples < 1000:
            n_splits = 5
        else:
            n_splits = 10

        tscv = TimeSeriesSplit(n_splits=n_splits)
        logger.info(f"Используется {n_splits} фолдов для кросс-валидации")

        # Адаптивные минимальные размеры
        min_val_size = max(20, n_samples // (n_splits * 4))
        min_train_size = max(30, n_samples // (n_splits * 2))
        
        best_score = 0
        best_model = None
        best_thr = 0.5
        scores = []
        thresholds = []
        metrics_log = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
            if len(val_idx) < min_val_size or len(train_idx) < min_train_size:
                logger.warning(f"Fold {fold + 1}: Недостаточно данных (train: {len(train_idx)}, val: {len(val_idx)}, "
                             f"минимум train: {min_train_size}, val: {min_val_size})")
                continue

            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if len(np.unique(y_train)) < 2:
                logger.warning(f"Fold {fold + 1}: Недостаточно классов в обучающей выборке")
                continue

            try:
                from imblearn.over_sampling import SMOTE
                min_class_samples = min(np.sum(y_train == 0), np.sum(y_train == 1))
                if min_class_samples > 5:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_samples - 1))
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                else:
                    logger.warning(f"Fold {fold + 1}: Недостаточно образцов для SMOTE (min_class: {min_class_samples})")
                    X_train_balanced, y_train_balanced = X_train, y_train
            except Exception as e:
                logger.warning(f"Fold {fold + 1}: Ошибка SMOTE: {e}, используем несбалансированные данные")
                X_train_balanced, y_train_balanced = X_train, y_train

            ensemble = EnsembleModel()
            try:
                ensemble.fit(X_train_balanced, y_train_balanced)
            except Exception as e:
                logger.error(f"Fold {fold + 1}: Ошибка обучения ансамбля: {e}")
                continue

            # Адаптивная калибровка
            if len(val_idx) > 50:  # Уменьшено со 100
                val_split = len(val_idx) // 2
                X_val_pred = X_val[:val_split]
                y_val_pred = y_val[:val_split]
                X_val_calib = X_val[val_split:]
                y_val_calib = y_val[val_split:]

                for name in ensemble.models:
                    try:
                        ensemble.models[name] = calibrate_model(ensemble.models[name], X_val_calib, y_val_calib)
                    except Exception as e:
                        logger.warning(f"Fold {fold + 1}: Ошибка калибровки {name}: {e}")

                y_pred_proba = ensemble.predict_proba(X_val_pred)[:, 1]
                y_val_eval = y_val_pred
            else:
                y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
                y_val_eval = y_val

            if np.any(np.isnan(y_pred_proba)):
                logger.warning(f"Fold {fold + 1}: Обнаружены NaN значения в предсказаниях, заменяем на 0.5")
                y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)

            if len(np.unique(y_val_eval)) > 1:
                try:
                    score = roc_auc_score(y_val_eval, y_pred_proba)
                    if score > 0.95:
                        logger.warning(f"Fold {fold + 1}: Подозрительно высокий AUC: {score:.4f}, возможно переобучение")

                    thr = find_best_threshold(y_val_eval, y_pred_proba)
                    thresholds.append(thr)
                    scores.append(score)
                    preds = (y_pred_proba > thr).astype(int)
                    f1 = f1_score(y_val_eval, preds)
                    prec = precision_score(y_val_eval, preds, zero_division=0)
                    rec = recall_score(y_val_eval, preds, zero_division=0)
                    mcc = matthews_corrcoef(y_val_eval, preds)
                    metrics_log.append((score, f1, prec, rec, mcc))
                    logger.info(f"Fold {fold + 1} AUC: {score:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, "
                              f"Recall: {rec:.4f}, MCC: {mcc:.4f}, Best threshold: {thr:.3f}")

                    if score > best_score:
                        best_score = score
                        best_model = ensemble

                except Exception as e:
                    logger.error(f"Fold {fold + 1}: Ошибка расчета метрик: {e}")
                    continue

        if scores and np.mean(scores) > 0.95:
            logger.warning("Средний AUC слишком высок, возможно переобучение. Проверьте данные на утечку информации.")

        if best_model is None:
            logger.warning("Не удалось найти лучшую модель, обучаем на всех данных")
            try:
                from imblearn.over_sampling import SMOTE
                min_class_samples = min(np.sum(y == 0), np.sum(y == 1))
                if min_class_samples > 5:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_samples - 1))
                    X_balanced, y_balanced = smote.fit_resample(X_selected, y)
                else:
                    logger.warning(f"Недостаточно образцов для SMOTE на всех данных (min_class: {min_class_samples})")
                    X_balanced, y_balanced = X_selected, y
            except Exception as e:
                logger.warning(f"Ошибка SMOTE на всех данных: {e}")
                X_balanced, y_balanced = X_selected, y

            best_model = EnsembleModel()
            best_model.fit(X_balanced, y_balanced)

        avg_score = np.mean(scores) if scores else 0.5
        avg_thr = np.mean(thresholds) if thresholds else 0.5
        logger.info(f"Модель обучена. Средний AUC: {avg_score:.4f}, Лучший AUC: {best_score:.4f}, "
                   f"Средний порог: {avg_thr:.3f}, Успешных фолдов: {len(scores)}/{n_splits}")

        try:
            if 'xgb' in best_model.models and len(X_selected) > 100:
                sample_size = min(1000, len(X_selected))
                sample_indices = np.random.choice(len(X_selected), sample_size, replace=False)
                shap_feature_importance(best_model.models['xgb'], X_selected[sample_indices], selected_features)
        except Exception as e:
            logger.warning(f"SHAP-анализ не выполнен: {e}")

        feature_importance = best_model.get_feature_importance(selected_features)
        if feature_importance is not None:
            logger.info(f"Топ-10 важных признаков:\n{feature_importance.head(10)}")

        return best_model, scaler, selected_features, avg_thr, all_feature_columns

    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        logger.error(traceback.format_exc())
        return None, None, [], 0.5, []


def predict_enhanced_direction(
    df: pd.DataFrame,
    model: EnsembleModel,
    scaler,
    features: List[str],
    commission: float,
    threshold: float = 0.5,
    all_features: List[str] = None  # Добавляем параметр для всех признаков
) -> Tuple[int, float]:
    from calculations import prepare_enhanced_features

    if df.empty or model is None or scaler is None or not features:
        logger.warning("[predict_enhanced_direction] Пустые входные данные")
        return 0, 0.5

    try:
        # Используем ту же функцию, что и при обучении, но с параметром for_prediction=True
        df_feat = prepare_enhanced_features(df, commission, for_prediction=True)
        if df_feat.empty or len(df_feat) < 1:
            logger.warning("[predict_enhanced_direction] Пустой DataFrame после обработки индикаторов")
            return 0, 0.5

        # Получаем последнюю строку
        latest_row = df_feat.iloc[-1]
        
        # Если all_features не передан, пытаемся восстановить из данных
        if all_features is None:
            all_features = [col for col in df_feat.columns if col not in ['target', 'target_multi', 'future_return', 'max_future_return']]
            logger.warning("all_features не передан, используем признаки из текущих данных")
        
        # Создаем массив со ВСЕМИ признаками для масштабирования
        X_full = np.zeros((1, len(all_features)))
        
        # Заполняем массив доступными значениями
        for i, feature in enumerate(all_features):
            if feature in latest_row.index:
                X_full[0, i] = latest_row[feature]
            else:
                X_full[0, i] = 0  # Заполняем нулями отсутствующие признаки
        
        # Масштабируем ВСЕ признаки
        X_scaled_full = scaler.transform(X_full)
        
        # Теперь выбираем только те признаки, которые были отобраны BorutaPy
        # Находим индексы отобранных признаков в полном списке
        selected_indices = []
        for selected_feature in features:
            if selected_feature in all_features:
                idx = all_features.index(selected_feature)
                selected_indices.append(idx)
            else:
                logger.warning(f"Признак {selected_feature} не найден в списке всех признаков")
        
        # Создаем массив только с отобранными признаками
        if selected_indices:
            X_selected = X_scaled_full[:, selected_indices]
        else:
            logger.error("Не найдено ни одного отобранного признака")
            return 0, 0.5
        
        # Проверка размерности
        if X_selected.shape[1] != len(features):
            logger.error(f"Несоответствие размерности: ожидалось {len(features)}, получено {X_selected.shape[1]}")
            return 0, 0.5
        
        # Проверка на NaN после масштабирования
        if np.isnan(X_selected).any():
            logger.warning("Обнаружены NaN значения после масштабирования, заменяем на 0")
            X_selected = np.nan_to_num(X_selected, nan=0.0)
        
        # Предсказание на отобранных признаках
        proba = model.predict_proba(X_selected)[0][1]
        
        # Проверка на NaN в вероятности
        if np.isnan(proba):
            logger.warning("Получена NaN вероятность, используем 0.5")
            proba = 0.5
            
        label = int(proba > threshold)
        return label, proba

    except Exception as e:
        logger.error(f"[predict_enhanced_direction] Ошибка: {e}")
        logger.error(traceback.format_exc())
        return 0, 0.5

# --- Адаптивный менеджер порогов ---

class AdaptiveThresholdManager:
    def __init__(self, window: int = 100):
        self.window = window
        self.threshold_history = deque(maxlen=window)
        self.performance_history = deque(maxlen=window)

    def calculate_threshold(self, market_data, market_conditions, base_threshold: float = 0.52) -> float:
        volatility_adjustment = market_conditions.volatility_percentile / 2000
        regime_adjustments = {
            "high_volatility": 0.02,
            "trending_up": -0.02,
            "trending_down": -0.02,
            "ranging": 0.01,
            "unknown": 0.00
        }
        regime_adjustment = regime_adjustments.get(market_conditions.regime, 0)

        performance_adjustment = 0
        if len(self.performance_history) >= 20:
            recent_accuracy = np.mean(list(self.performance_history)[-20:])
            if recent_accuracy < 0.45:
                performance_adjustment = 0.01
            elif recent_accuracy > 0.65:
                performance_adjustment = -0.01

        current_hour = datetime.datetime.now().hour
        time_adjustment = 0
        if current_hour in [14, 15, 16]:
            time_adjustment = 0.005
        elif current_hour in [2, 3, 4]:
            time_adjustment = 0.01

        final_threshold = base_threshold + volatility_adjustment + regime_adjustment + performance_adjustment + time_adjustment
        return np.clip(final_threshold, 0.50, 0.65)

    def update_performance(self, predicted: bool, actual: bool):
        self.performance_history.append(predicted == actual)


# --- Улучшенный анализатор сигналов ---

class EnhancedSignalAnalyzer:
    def __init__(self, commission: float, spread: float):
        self.commission = commission
        self.spread = spread
        self.signal_history = deque(maxlen=100)
        self.microstructure_weight = 0.3  # Вес микроструктурных данных в общем сигнале

    def analyze_signal(self, df: pd.DataFrame, prob_up: float, market_conditions,
                       confidence_score: float, microstructure_data: dict = None) -> Tuple[str, float, str, float]:
        try:
            if df.empty:
                logger.warning("Получен пустой DataFrame для анализа сигналов")
                return "HOLD", 0.5, "Недостаточно данных", 0.0

            last = df.iloc[-1]
            signals = []
            signal_strengths = []

            trend_signal, trend_strength = self._analyze_trend(df, last)
            signals.append(trend_signal)
            signal_strengths.append(trend_strength * 0.8)

            momentum_signal, momentum_strength = self._analyze_momentum(df, last)
            signals.append(momentum_signal)
            signal_strengths.append(momentum_strength)

            volume_signal, volume_strength = self._analyze_volume(df, last)
            signals.append(volume_signal)
            signal_strengths.append(volume_strength * 0.7)

            pattern_signal, pattern_strength = self._analyze_patterns(df, last)
            signals.append(pattern_signal)
            signal_strengths.append(pattern_strength * 0.6)

            ml_signal = 1 if prob_up > 0.5 else -1
            ml_strength = abs(prob_up - 0.5) * 3
            signals.append(ml_signal)
            signal_strengths.append(ml_strength)

            if microstructure_data:
                micro_signal, micro_strength = self._analyze_microstructure(microstructure_data)
                signals.append(micro_signal)
                signal_strengths.append(micro_strength * self.microstructure_weight)

            weighted_signal = np.average(signals, weights=signal_strengths)
            overall_strength = np.mean(signal_strengths)

            if market_conditions.regime == "high_volatility":
                threshold_multiplier = 1.2
            elif market_conditions.regime in ["trending_up", "trending_down"]:
                threshold_multiplier = 0.7
            else:
                threshold_multiplier = 1.0

            signal_threshold = 0.15 * threshold_multiplier

            if weighted_signal > signal_threshold and overall_strength > 0.25:
                signal = "BUY"
                reason = self._generate_reason(signals, signal_strengths, "BUY", market_conditions, microstructure_data)
            elif weighted_signal < -signal_threshold and overall_strength > 0.25:
                signal = "SELL"
                reason = self._generate_reason(signals, signal_strengths, "SELL", market_conditions, microstructure_data)
            else:
                signal = "HOLD"
                reason = f"Слабый сигнал: {weighted_signal:.2f}, сила: {overall_strength:.2f}"

            final_confidence = confidence_score * overall_strength

            self.signal_history.append({
                'timestamp': datetime.datetime.now(),
                'signal': signal,
                'weighted_signal': weighted_signal,
                'strength': overall_strength,
                'confidence': final_confidence,
                'market_regime': market_conditions.regime,
                'microstructure': microstructure_data is not None
            })

            return signal, overall_strength, reason, final_confidence

        except Exception as e:
            logger.error(f"Ошибка анализа сигналов: {e}")
            logger.error(traceback.format_exc())
            return "HOLD", 0.0, f"Ошибка анализа: {e}", 0.0

    def _analyze_trend(self, df: pd.DataFrame, last: pd.Series) -> Tuple[int, float]:
        signal = 0
        strength = 0

        if 'ma_20' in last and 'ma_50' in last:
            if last['ma_20'] > last['ma_50']:
                signal += 1
                strength += abs(last['ma_20'] - last['ma_50']) / last['close']
            else:
                signal -= 1
                strength += abs(last['ma_20'] - last['ma_50']) / last['close']

        if 'adx' in df.columns and last.get('adx', 0) > 25:
            strength *= 1.5

        if 'macd_hist' in last:
            if last['macd_hist'] > 0:
                signal += 1
            else:
                signal -= 1
            strength += abs(last['macd_hist']) / last['close'] * 100

        return np.sign(signal), min(strength, 1.0)

    def _analyze_momentum(self, df: pd.DataFrame, last: pd.Series) -> Tuple[int, float]:
        signal = 0
        strength = 0

        if 'rsi' in last:
            if last['rsi'] < 30:
                signal += 2
                strength += (30 - last['rsi']) / 30
            elif last['rsi'] > 70:
                signal -= 2
                strength += (last['rsi'] - 70) / 30
            else:
                strength += 0.3

        if 'stoch_k' in last and 'stoch_d' in last:
            if last['stoch_k'] < 20 and last['stoch_k'] > last['stoch_d']:
                signal += 1
                strength += 0.3
            elif last['stoch_k'] > 80 and last['stoch_k'] < last['stoch_d']:
                signal -= 1
                strength += 0.3

        if 'mfi' in last:
            if last['mfi'] < 20:
                signal += 1
                strength += 0.2
            elif last['mfi'] > 80:
                signal -= 1
                strength += 0.2

        return np.sign(signal), min(strength, 1.0)

    def _analyze_volume(self, df: pd.DataFrame, last: pd.Series) -> Tuple[int, float]:
        signal = 0
        strength = 0

        if 'volume_ratio' in last:
            if last['volume_ratio'] > 1.5:
                strength += 0.5
                if 'return_1m' in last and last['return_1m'] > 0:
                    signal += 1
                else:
                    signal -= 1

        if 'obv_slope' in last:
            if last['obv_slope'] > 0:
                signal += 1
                strength += min(abs(last['obv_slope']) * 10, 0.5)
            else:
                signal -= 1
                strength += min(abs(last['obv_slope']) * 10, 0.5)

        return np.sign(signal), min(strength, 1.0)

    def _analyze_patterns(self, df: pd.DataFrame, last: pd.Series) -> Tuple[int, float]:
        signal = 0
        strength = 0

        if last.get('hammer', 0) == 1:
            signal += 2
            strength += 0.7
        elif last.get('shooting_star', 0) == 1:
            signal -= 2
            strength += 0.7

        if 'bb_position' in last:
            if last['bb_position'] < 0.2:
                signal += 1
                strength += 0.4
            elif last['bb_position'] > 0.8:
                signal -= 1
                strength += 0.4

        if 'price_to_vwap' in last:
            if last['price_to_vwap'] < 0.98:
                signal += 1
                strength += 0.3
            elif last['price_to_vwap'] > 1.02:
                signal -= 1
                strength += 0.3

        return np.sign(signal), min(strength, 1.0)

    def _analyze_microstructure(self, microstructure_data: dict) -> Tuple[int, float]:
        signal = 0
        strength = 0

        imbalance = microstructure_data.get('imbalance', 0)
        if imbalance > 0.2:
            signal += 1
            strength += min(abs(imbalance), 0.5)
        elif imbalance < -0.2:
            signal -= 1
            strength += min(abs(imbalance), 0.5)

        imbalance_trend = microstructure_data.get('imbalance_trend', 0)
        if imbalance_trend > 0.05:
            signal += 1
            strength += min(abs(imbalance_trend) * 5, 0.3)
        elif imbalance_trend < -0.05:
            signal -= 1
            strength += min(abs(imbalance_trend) * 5, 0.3)

        pressure_ratio = microstructure_data.get('pressure_ratio', 1)
        if pressure_ratio > 1.2:
            signal += 1
            strength += min((pressure_ratio - 1) * 0.5, 0.4)
        elif pressure_ratio < 0.8:
            signal -= 1
            strength += min((1 - pressure_ratio) * 0.5, 0.4)

        cvd_normalized = microstructure_data.get('cvd_normalized', 0)
        if cvd_normalized > 0.1:
            signal += 1
            strength += min(abs(cvd_normalized) * 2, 0.4)
        elif cvd_normalized < -0.1:
            signal -= 1
            strength += min(abs(cvd_normalized) * 2, 0.4)

        spread_pct = microstructure_data.get('spread_pct', 0)
        spread_trend = microstructure_data.get('spread_trend', 0)
        if spread_pct > 0.1 and spread_trend > 0:
            strength *= 0.8

        if microstructure_data.get('price_jump', False):
            strength *= 0.7

        return np.sign(signal), min(strength, 1.0)

    def _generate_reason(self, signals: List[int], strengths: List[float], final_signal: str,
                         market_conditions, microstructure_data: dict = None) -> str:
        components = []
        signal_names = ["Тренд", "Моментум", "Объем", "Паттерны", "ML"]

        if microstructure_data:
            signal_names.append("Микроструктура")

        for i, (sig, strength, name) in enumerate(zip(signals, strengths, signal_names)):
            if strength > 0.2:
                direction = "↑" if sig > 0 else "↓" if sig < 0 else "→"
                components.append(f"{name}{direction}({strength:.1f})")

        reason = f"Режим: {market_conditions.regime}, Сигналы: {', '.join(components)}"

        if microstructure_data:
            micro_details = []

            if abs(microstructure_data.get('imbalance', 0)) > 0.2:
                imb_dir = "покупателей" if microstructure_data.get('imbalance', 0) > 0 else "продавцов"
                micro_details.append(f"Дисбаланс {imb_dir}")

            if abs(microstructure_data.get('cvd_normalized', 0)) > 0.1:
                cvd_dir = "покупки" if microstructure_data.get('cvd_normalized', 0) > 0 else "продажи"
                micro_details.append(f"Поток {cvd_dir}")

            if microstructure_data.get('price_jump', False):
                micro_details.append("Скачок цены")

            if micro_details:
                reason += f", Микро: {', '.join(micro_details)}"

        return reason

# --- Улучшенный бэктестер ---

class Backtester:
    def __init__(self, initial_balance: float = 10000, commission: float = 0.001):
        self.initial_balance = initial_balance
        self.commission = commission
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.in_position = False
        self.current_position = None
        self.drawdowns = []
        self.current_drawdown = 0
        self.peak_balance = self.initial_balance

    def backtest_strategy(self, df: pd.DataFrame, model, scaler, features,
                          signal_analyzer: 'EnhancedSignalAnalyzer',
                          market_detector, threshold_manager: 'AdaptiveThresholdManager',
                          threshold: float = 0.5,
                          all_features: List[str] = None) -> dict:
        from calculations import calculate_enhanced_indicators, prepare_enhanced_features

        self.reset()
        min_lookback = 200

        if df.empty or len(df) < min_lookback:
            logger.error("Недостаточно данных для бэктеста")
            return self._get_empty_results()

        df_backtest = df.copy()
        timestamps = df_backtest.index.copy()
        logger.info("Предварительный расчет индикаторов для всего датасета...")
        df_with_indicators = calculate_enhanced_indicators(df_backtest)

        if df_with_indicators.empty:
            logger.error("Не удалось рассчитать индикаторы")
            return self._get_empty_results()

        logger.info("Подготовка признаков для бэктеста...")
        step = 1
        indices_to_process = list(range(min_lookback, len(df_backtest), step))
        prepared_data = {}
        batch_size = 50

        for batch_start in range(0, len(indices_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(indices_to_process))
            batch_indices = indices_to_process[batch_start:batch_end]

            if batch_start % 200 == 0:
                logger.info(f"Подготовка признаков: {batch_start}/{len(indices_to_process)}")

            for idx in batch_indices:
                current_data = df_with_indicators.iloc[:idx+1]
                features_df = prepare_enhanced_features(
                    current_data,
                    self.commission,
                    for_prediction=True
                )

                if not features_df.empty:
                    prepared_data[idx] = {
                        'features': features_df,
                        'price': df_backtest['close'].iloc[idx],
                        'timestamp': timestamps[idx],
                        'market_data': current_data
                    }

        logger.info(f"Запуск бэктеста для {len(prepared_data)} точек...")

        # Цикл по prepared_data пойдет в следующей части
        for idx in indices_to_process:
            try:
                if idx not in prepared_data:
                    continue

                data = prepared_data[idx]
                current_price = data['price']
                current_timestamp = data['timestamp']
                current_data_with_indicators = data['market_data']
                features_df = data['features']

                market_conditions = market_detector.detect_regime(current_data_with_indicators)

                try:
                    if all_features:
                        X = features_df[all_features].iloc[-1:].fillna(0)
                    else:
                        available_features = [f for f in features if f in features_df.columns]
                        X = features_df[available_features].iloc[-1:].fillna(0)

                    X_scaled = scaler.transform(X)

                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_scaled)[0]
                        prob_up = prob[1] if len(prob) > 1 else prob[0]
                    else:
                        prediction = model.predict(X_scaled)[0]
                        prob_up = prediction

                except Exception as e:
                    logger.debug(f"Ошибка предсказания: {e}")
                    continue

                threshold_val = threshold_manager.calculate_threshold(current_data_with_indicators, market_conditions)
                confidence_score = abs(prob_up - 0.5) * 2
                signal, strength, reason, confidence = signal_analyzer.analyze_signal(
                    current_data_with_indicators, prob_up, market_conditions, confidence_score
                )

                if not self.in_position:
                    if signal == "BUY" and confidence > 0.3:
                        self._open_position("LONG", current_price, current_timestamp, market_conditions, confidence)
                    elif signal == "SELL" and confidence > 0.3:
                        self._open_position("SHORT", current_price, current_timestamp, market_conditions, confidence)
                else:
                    should_close, close_reason = self._check_close_conditions(
                        current_price, signal, market_conditions
                    )
                    if should_close:
                        self._close_position(current_price, current_timestamp, close_reason)

                current_equity = self._calculate_equity(current_price)
                self.equity_curve.append(current_equity)

                if current_equity > self.peak_balance:
                    self.peak_balance = current_equity
                    self.current_drawdown = 0
                else:
                    self.current_drawdown = (self.peak_balance - current_equity) / self.peak_balance * 100
                    self.drawdowns.append(self.current_drawdown)

            except Exception as e:
                logger.error(f"Ошибка в бэктесте на индексе {idx}: {e}")
                continue

        if self.in_position:
            last_price = df_backtest['close'].iloc[-1]
            last_timestamp = timestamps[-1]
            self._close_position(last_price, last_timestamp, "End of backtest")

        return self._calculate_backtest_metrics()
    
    def _open_position(self, side: str, price: float, timestamp, market_conditions, confidence: float):
        position_size_usd = self.balance * 0.02
        position_size = position_size_usd / price
        commission_cost = position_size * price * self.commission
        self.balance -= commission_cost

        self.current_position = {
            'side': side,
            'entry_price': price,
            'size': position_size,
            'entry_time': timestamp,
            'market_regime': market_conditions.regime,
            'confidence': confidence,
            'max_profit': 0.0,
            'max_loss': 0.0
        }

        self.in_position = True
        logger.debug(f"Открыта позиция {side} по цене {price} ({timestamp})")

    def _check_close_conditions(self, current_price: float, signal: str, market_conditions) -> Tuple[bool, str]:
        if not self.current_position:
            return False, ""

        if self.current_position['side'] == "LONG":
            profit_pct = (current_price - self.current_position['entry_price']) / self.current_position['entry_price']
        else:
            profit_pct = (self.current_position['entry_price'] - current_price) / self.current_position['entry_price']

        if profit_pct > 0:
            self.current_position['max_profit'] = max(self.current_position['max_profit'], profit_pct)
        else:
            self.current_position['max_loss'] = min(self.current_position['max_loss'], profit_pct)

        if profit_pct < -0.02:
            return True, "Stop Loss"
        if profit_pct > 0.03:
            return True, "Take Profit"

        if (self.current_position['side'] == "LONG" and signal == "SELL") or \
           (self.current_position['side'] == "SHORT" and signal == "BUY"):
            return True, "Reverse Signal"

        if self.current_position['max_profit'] > 0.015 and \
           (self.current_position['max_profit'] - profit_pct) > 0.005:
            return True, "Trailing Stop"

        if market_conditions.regime != self.current_position['market_regime'] and profit_pct > 0:
            return True, "Market Regime Change"

        return False, ""
    
    def _close_position(self, exit_price: float, exit_time, reason: str):
        if not self.current_position:
            return

        if self.current_position['side'] == "LONG":
            gross_profit = self.current_position['size'] * (exit_price - self.current_position['entry_price'])
        else:
            gross_profit = self.current_position['size'] * (self.current_position['entry_price'] - exit_price)

        commission_cost = self.current_position['size'] * exit_price * self.commission
        net_profit = gross_profit - commission_cost
        self.balance += (self.current_position['size'] * exit_price - commission_cost)

        profit_pct = net_profit / (self.current_position['size'] * self.current_position['entry_price']) * 100

        trade = {
            'side': self.current_position['side'],
            'entry_price': self.current_position['entry_price'],
            'exit_price': exit_price,
            'size': self.current_position['size'],
            'profit': net_profit,
            'profit_pct': profit_pct,
            'entry_time': self.current_position['entry_time'],
            'exit_time': exit_time,
            'duration': (exit_time - self.current_position['entry_time']).total_seconds() / 60,
            'market_regime': self.current_position['market_regime'],
            'confidence': self.current_position['confidence'],
            'max_profit': self.current_position['max_profit'] * 100,
            'max_loss': self.current_position['max_loss'] * 100,
            'reason': reason
        }

        self.trades.append(trade)
        self.in_position = False
        self.current_position = None

        logger.debug(f"Закрыта позиция по цене {exit_price}, прибыль: {net_profit:.2f} ({profit_pct:.2f}%), причина: {reason}")

    def _calculate_equity(self, current_price: float) -> float:
        equity = self.balance

        if self.in_position and self.current_position:
            if self.current_position['side'] == "LONG":
                unrealized = self.current_position['size'] * (current_price - self.current_position['entry_price'])
            else:
                unrealized = self.current_position['size'] * (self.current_position['entry_price'] - current_price)

            unrealized -= self.current_position['size'] * current_price * self.commission
            equity += unrealized

        return equity
    
    def _calculate_backtest_metrics(self) -> dict:
        if not self.trades:
            return self._get_empty_results()

        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        profits = [t['profit'] for t in self.trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0.0001
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        max_drawdown = max(self.drawdowns) if self.drawdowns else 0

        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for profit in profits:
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

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
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def _get_empty_results(self) -> dict:
        return {
            'total_return': 0,
            'final_balance': self.initial_balance,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'trades': [],
            'equity_curve': [self.initial_balance]
        }
    
    def visualize_backtest_results(self, symbol: str = "Symbol") -> Tuple[plt.Figure, str]:
        if not self.trades:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, "Нет данных для визуализации",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            return fig, "Нет данных для визуализации"

        fig = plt.figure(figsize=(15, 12))
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax1.plot(self.equity_curve, linewidth=2)
        ax1.set_title(f'Кривая эквити - {symbol}')
        ax1.set_ylabel('Баланс (USDT)')
        ax1.grid(True)

        # Просадка
        ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max * 100
        ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
        ax2.plot(drawdown, color='red', linewidth=1.5)
        ax2.set_title('Просадка (%)')
        ax2.set_ylabel('Просадка (%)')
        ax2.set_xlabel('Время (периоды)')
        ax2.grid(True)

        # Распределение прибыли
        ax3 = plt.subplot2grid((3, 2), (2, 0))
        profits = [t['profit'] for t in self.trades]
        profits_positive = [p for p in profits if p > 0]
        profits_negative = [p for p in profits if p < 0]
        bins = np.linspace(min(profits), max(profits), 20)
        ax3.hist(profits_positive, bins=bins, alpha=0.7, color='green', label='Прибыль')
        ax3.hist(profits_negative, bins=bins, alpha=0.7, color='red', label='Убыток')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax3.set_title('Распределение прибыли')
        ax3.set_xlabel('Прибыль (USDT)')
        ax3.set_ylabel('Количество сделок')
        ax3.legend()

        # Производительность по режимам
        ax4 = plt.subplot2grid((3, 2), (2, 1))
        regime_performance = {}
        for trade in self.trades:
            regime = trade.get('market_regime', 'unknown')
            if regime not in regime_performance:
                regime_performance[regime] = {'count': 0, 'profit': 0, 'wins': 0}
            regime_performance[regime]['count'] += 1
            regime_performance[regime]['profit'] += trade['profit']
            if trade['profit'] > 0:
                regime_performance[regime]['wins'] += 1

        regimes = list(regime_performance.keys())
        profits_by_regime = [regime_performance[r]['profit'] for r in regimes]
        colors = ['green' if p > 0 else 'red' for p in profits_by_regime]
        bars = ax4.bar(regimes, profits_by_regime, alpha=0.7, color=colors)

        for bar, regime in zip(bars, regimes):
            height = bar.get_height()
            count = regime_performance[regime]['count']
            wins = regime_performance[regime]['wins']
            win_rate = wins / count if count > 0 else 0
            ax4.text(bar.get_x() + bar.get_width() / 2., height + (max(profits_by_regime) * 0.02),
                     f'{count} сделок\n{win_rate:.0%}', ha='center', va='bottom', fontsize=9)

        ax4.set_title('Прибыль по режимам рынка')
        ax4.set_xlabel('Режим рынка')
        ax4.set_ylabel('Прибыль (USDT)')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        report = f"📊 <b>Отчет по стратегии для {symbol}</b>\n\n"
        report += f"• Всего сделок: {len(self.trades)}\n"
        report += f"• Финальный баланс: {self.balance:.2f} USDT\n"
        report += f"• Чистая прибыль: {self.balance - self.initial_balance:.2f} USDT\n"
        report += f"• Доходность: {(self.balance - self.initial_balance) / self.initial_balance * 100:.2f}%\n"
        report += f"• Win rate: {np.mean([1 if t['profit'] > 0 else 0 for t in self.trades]):.2%}\n"
        report += f"• Max drawdown: {max(self.drawdowns) if self.drawdowns else 0:.2f}%\n"
        report += f"• Sharpe: {self._calculate_backtest_metrics().get('sharpe_ratio', 0):.2f}\n"

        return fig, report
    
    def validate_dataframe_for_trading(
        df: pd.DataFrame,
        required_columns: List[str] = None
    ) -> Tuple[bool, str]:
        """
        Валидация DataFrame для использования в модели.

        :param df: Входной DataFrame с OHLCV
        :param required_columns: Список обязательных колонок
        :return: (bool, str) — прошла ли валидация и сообщение
        """
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Проверка на пустоту
        if df.empty:
            return False, "DataFrame пустой"

        # Проверка длины
        if len(df) < 200:
            return False, f"Недостаточно данных: {len(df)} строк (минимум 200)"

        # Проверка наличия колонок
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Отсутствуют колонки: {missing_columns}"

        # Проверка на NaN
        for col in required_columns:
            if df[col].isna().any():
                nan_count = df[col].isna().sum()
                return False, f"NaN в '{col}': {nan_count} значений"

        # Отрицательные / неположительные значения
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and (df[col] <= 0).any():
                return False, f"Некорректные значения в колонке '{col}'"

        if 'volume' in df.columns and (df['volume'] < 0).any():
            return False, "Обнаружены отрицательные значения в объеме"

        # Проверка индекса
        if not isinstance(df.index, pd.DatetimeIndex):
            return False, "Индекс должен быть типа datetime"

        if df.index.duplicated().any():
            return False, f"Дубликаты во временном индексе: {df.index.duplicated().sum()}"

        if not df.index.is_monotonic_increasing:
            return False, "Временной индекс должен быть монотонно возрастающим"

        return True, "Валидация пройдена успешно"
