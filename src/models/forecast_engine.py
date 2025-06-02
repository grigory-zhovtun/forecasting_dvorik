"""
Модуль прогнозирования с различными моделями
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import joblib
from pickle import PicklingError
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM модель для временных рядов"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ForecastEngine:
    """Основной класс для прогнозирования"""
    
    def __init__(self, config: dict, data_loader=None):
        """
        Инициализация движка прогнозирования
        
        Args:
            config: Словарь конфигурации
            data_loader: Экземпляр DataLoader
        """
        self.config = config
        self.data_loader = data_loader
        self.forecast_defaults = config.get('forecast_defaults', {})
        self.metrics_thresholds = config.get('metrics_thresholds', {})
        self.auto_tuning_config = config.get('auto_tuning', {})
        
        # Загружаем праздники
        self.holidays = data_loader.load_holidays()
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик качества прогноза
        
        Args:
            y_true: Фактические значения
            y_pred: Прогнозные значения
            
        Returns:
            Словарь с метриками
        """
        # Удаляем NaN значения
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {'MAPE': None, 'MAE': None, 'RMSE': None, 'R2': None}
        
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAPE': round(mape, 2),
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4)
        }
    
    def prepare_time_series_data(self, data: np.ndarray, lookback: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка данных для моделей временных рядов
        
        Args:
            data: Временной ряд
            lookback: Количество предыдущих точек для предсказания
            
        Returns:
            X, y для обучения
        """
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def auto_tune_prophet(self, df: pd.DataFrame, metric: str, params: dict, n_trials: int = 50) -> dict:
        """Автоматическая настройка гиперпараметров Prophet"""
        # Подготовка данных
        df_model = df[['Дата', metric]].rename(columns={'Дата': 'ds', metric: 'y'})
        
        # Разделение на обучающую и тестовую выборки
        train_split = params.get('train_split', 0.8)
        train_size = int(len(df_model) * train_split)
        train_df = df_model.iloc[:train_size]
        test_df = df_model.iloc[train_size:]
        
        if len(test_df) < 30:
            test_df = df_model.iloc[-30:]
        
        def objective(trial):
            # Гиперпараметры для оптимизации
            changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
            seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True)
            holidays_prior_scale = trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True)
            
            # Обучение модели
            model = Prophet(
                holidays=self.holidays,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                yearly_seasonality=params.get('yearly_seasonality', True),
                weekly_seasonality=params.get('weekly_seasonality', True),
                daily_seasonality=params.get('daily_seasonality', False)
            )
            
            model.fit(train_df)
            
            # Прогноз
            future = model.make_future_dataframe(periods=len(test_df))
            forecast = model.predict(future)
            
            # Выбираем только даты из тестовой выборки
            test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
            merged = test_forecast.merge(test_df, on='ds')
            
            # Считаем MAPE
            mape = mean_absolute_percentage_error(merged['y'], merged['yhat']) * 100
            
            return mape
        
        # Запускаем оптимизацию
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, TypeError, PicklingError) as e:
            logger.warning(f"Параллельная оптимизация не удалась: {e}. Используем n_jobs=1")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        best_params = study.best_params
        best_mape = study.best_value
        
        # Обновляем параметры
        params.update(best_params)
        
        return {'params': best_params, 'mape': best_mape}
    
    def auto_tune_xgboost(self, df: pd.DataFrame, metric: str, params: dict, n_trials: int = 50) -> dict:
        """Автоматическая настройка гиперпараметров XGBoost"""
        # Подготовка признаков
        df_features = df.copy()
        df_features['day_of_week'] = df_features['Дата'].dt.dayofweek
        df_features['month'] = df_features['Дата'].dt.month
        df_features['day'] = df_features['Дата'].dt.day
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Создаем лаговые признаки
        for lag in [1, 7, 14, 30]:
            df_features[f'lag_{lag}'] = df_features[metric].shift(lag)
        
        df_features = df_features.dropna()
        
        # Определяем признаки
        features = ['day_of_week', 'month', 'day', 'is_weekend'] + [f'lag_{i}' for i in [1, 7, 14, 30]]
        
        # Разделение на обучающую и тестовую выборки
        train_split = params.get('train_split', 0.8)
        train_size = int(len(df_features) * train_split)
        
        X = df_features[features]
        y = df_features[metric]
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        def objective(trial):
            # Гиперпараметры для оптимизации
            n_estimators = trial.suggest_int('n_estimators', 50, 500)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            
            # Обучение модели
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Прогноз и метрика
            y_pred = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            return mape
        
        # Запускаем оптимизацию
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, TypeError, PicklingError) as e:
            logger.warning(f"Параллельная оптимизация не удалась: {e}. Используем n_jobs=1")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        best_params = study.best_params
        best_mape = study.best_value
        
        # Обновляем параметры для модели
        params.update({
            'xgb_n_estimators': best_params['n_estimators'],
            'xgb_max_depth': best_params['max_depth'],
            'xgb_learning_rate': best_params['learning_rate'],
            'xgb_subsample': best_params['subsample'],
            'xgb_colsample_bytree': best_params['colsample_bytree']
        })
        
        return {'params': best_params, 'mape': best_mape}
    
    def auto_tune_arima(self, df: pd.DataFrame, metric: str, params: dict, n_trials: int = 30) -> dict:
        """Автоматическая настройка гиперпараметров ARIMA"""
        data = df.set_index('Дата')[metric]
        
        # Разделение на обучающую и тестовую выборки
        train_split = params.get('train_split', 0.8)
        train_size = int(len(data) * train_split)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        def objective(trial):
            # Гиперпараметры для оптимизации
            p = trial.suggest_int('p', 0, 5)
            d = trial.suggest_int('d', 0, 2)
            q = trial.suggest_int('q', 0, 5)
            
            # Проверка на допустимость комбинации параметров
            if p == 0 and d == 0 and q == 0:
                return float('inf')
            
            try:
                # Обучение модели
                model = ARIMA(train_data, order=(p, d, q))
                model_fit = model.fit()
                
                # Прогноз
                forecast = model_fit.forecast(steps=len(test_data))
                
                # Метрика
                mape = mean_absolute_percentage_error(test_data, forecast) * 100
                return mape
            except:
                return float('inf')
        
        # Запускаем оптимизацию
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, TypeError, PicklingError) as e:
            logger.warning(f"Параллельная оптимизация не удалась: {e}. Используем n_jobs=1")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        best_params = study.best_params
        best_mape = study.best_value
        
        # Обновляем параметры
        params.update({
            'arima_p': best_params['p'],
            'arima_d': best_params['d'],
            'arima_q': best_params['q']
        })
        
        return {'params': best_params, 'mape': best_mape}
    
    def auto_tune_lstm(self, df: pd.DataFrame, metric: str, params: dict, n_trials: int = 20) -> dict:
        """Автоматическая настройка гиперпараметров LSTM"""
        # Нормализация данных
        scaler = StandardScaler()
        data = df[metric].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)
        
        # Параметры
        lookback = params.get('lstm_lookback', 30)
        
        # Подготовка данных
        X, y = self.prepare_time_series_data(df[metric].values, lookback)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Разделение на обучающую и тестовую выборки
        train_split = params.get('train_split', 0.8)
        train_size = int(len(X) * train_split)
        
        X_train, y_train = torch.FloatTensor(X[:train_size]), torch.FloatTensor(y[:train_size])
        X_test, y_test = torch.FloatTensor(X[train_size:]), torch.FloatTensor(y[train_size:])
        
        def objective(trial):
            # Гиперпараметры для оптимизации
            hidden_size = trial.suggest_int('hidden_size', 10, 100)
            num_layers = trial.suggest_int('num_layers', 1, 3)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Создание модели
            model = LSTMModel(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1
            )
            
            # Обучение
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            epochs = 20  # Уменьшаем для ускорения оптимизации
            
            for epoch in range(epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Оценка на тестовых данных
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test).squeeze().numpy()
                mape = mean_absolute_percentage_error(y_test.numpy(), y_pred) * 100
            
            return mape
        
        # Запускаем оптимизацию
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=1)  # LSTM плохо работает с n_jobs > 1
        except Exception as e:
            logger.error(f"Ошибка оптимизации LSTM: {e}")
            return {'params': {}, 'mape': float('inf')}
        
        best_params = study.best_params
        best_mape = study.best_value
        
        # Обновляем параметры
        params.update({
            'lstm_hidden_size': best_params['hidden_size'],
            'lstm_num_layers': best_params['num_layers'],
            'lstm_learning_rate': best_params['learning_rate'],
            'lstm_batch_size': best_params['batch_size']
        })
        
        return {'params': best_params, 'mape': best_mape}
    
    def forecast_prophet(self, df: pd.DataFrame, metric: str, params: dict) -> pd.DataFrame:
        """Прогнозирование с помощью Prophet"""
        df_model = df[['Дата', metric]].rename(columns={'Дата': 'ds', metric: 'y'})
        
        # Получаем параметры из конфига
        prophet_params = self.forecast_defaults.get('prophet', {})
        
        m = Prophet(
            holidays=self.holidays,
            changepoint_prior_scale=params.get('changepoint_prior_scale', prophet_params.get('changepoint_prior_scale', 0.05)),
            seasonality_prior_scale=params.get('seasonality_prior_scale', prophet_params.get('seasonality_prior_scale', 10.0)),
            holidays_prior_scale=params.get('holidays_prior_scale', prophet_params.get('holidays_prior_scale', 10.0)),
            yearly_seasonality=params.get('yearly_seasonality', prophet_params.get('yearly_seasonality', True)),
            weekly_seasonality=params.get('weekly_seasonality', prophet_params.get('weekly_seasonality', True)),
            daily_seasonality=params.get('daily_seasonality', prophet_params.get('daily_seasonality', False)),
            interval_width=params.get('confidence_interval', self.forecast_defaults.get('confidence_interval', 0.95))
        )
        
        m.fit(df_model)
        
        future = m.make_future_dataframe(periods=params.get('forecast_horizon', self.forecast_defaults.get('horizon', 365)))
        forecast = m.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def forecast_arima(self, df: pd.DataFrame, metric: str, params: dict) -> pd.DataFrame:
        """Прогнозирование с помощью ARIMA"""
        data = df.set_index('Дата')[metric]
        
        # Получаем параметры из конфига
        arima_params = self.forecast_defaults.get('arima', {})
        
        order = (
            params.get('arima_p', arima_params.get('p', 2)),
            params.get('arima_d', arima_params.get('d', 1)),
            params.get('arima_q', arima_params.get('q', 2))
        )
        
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        
        # Прогноз
        forecast_steps = params.get('forecast_horizon', self.forecast_defaults.get('horizon', 365))
        forecast_results = model_fit.get_forecast(steps=forecast_steps)
        forecast_values = forecast_results.predicted_mean
        
        # Доверительные интервалы
        confidence_interval = params.get('confidence_interval', self.forecast_defaults.get('confidence_interval', 0.95))
        alpha = 1 - confidence_interval
        conf_int = forecast_results.conf_int(alpha=alpha)
        
        # Создаем даты для прогноза
        last_date = df['Дата'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps)
        
        # Создаем DataFrame с прогнозом
        forecast_df = pd.DataFrame({
            'ds': pd.concat([df['Дата'], pd.Series(future_dates)]),
            'yhat': pd.concat([pd.Series(data.values), forecast_values])
        })
        
        # Добавляем границы доверительного интервала
        yhat_lower = pd.Series([np.nan] * len(df))
        yhat_upper = pd.Series([np.nan] * len(df))
        
        yhat_lower = pd.concat([yhat_lower, conf_int.iloc[:, 0]]).reset_index(drop=True)
        yhat_upper = pd.concat([yhat_upper, conf_int.iloc[:, 1]]).reset_index(drop=True)
        
        forecast_df['yhat_lower'] = yhat_lower
        forecast_df['yhat_upper'] = yhat_upper
        
        return forecast_df
    
    def forecast_xgboost(self, df: pd.DataFrame, metric: str, params: dict) -> pd.DataFrame:
        """Прогнозирование с помощью XGBoost"""
        # Подготовка признаков
        df_features = df.copy()
        df_features['day_of_week'] = df_features['Дата'].dt.dayofweek
        df_features['month'] = df_features['Дата'].dt.month
        df_features['day'] = df_features['Дата'].dt.day
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Создаем лаговые признаки
        for lag in [1, 7, 14, 30]:
            df_features[f'lag_{lag}'] = df_features[metric].shift(lag)
        
        df_features = df_features.dropna()
        
        # Разделение на обучающую выборку
        train_size = int(len(df_features) * params.get('train_split', self.forecast_defaults.get('train_split', 0.8)))
        train = df_features[:train_size]
        
        features = ['day_of_week', 'month', 'day', 'is_weekend'] + [f'lag_{i}' for i in [1, 7, 14, 30]]
        
        X_train = train[features]
        y_train = train[metric]
        
        # Получаем параметры из конфига
        xgb_params = self.forecast_defaults.get('xgboost', {})
        
        # Обучение модели
        model = xgb.XGBRegressor(
            n_estimators=params.get('xgb_n_estimators', xgb_params.get('n_estimators', 100)),
            max_depth=params.get('xgb_max_depth', xgb_params.get('max_depth', 5)),
            learning_rate=params.get('xgb_learning_rate', xgb_params.get('learning_rate', 0.1)),
            subsample=params.get('xgb_subsample', xgb_params.get('subsample', 0.8)),
            colsample_bytree=params.get('xgb_colsample_bytree', xgb_params.get('colsample_bytree', 0.8)),
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Прогнозирование
        last_date = df['Дата'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=params.get('forecast_horizon', self.forecast_defaults.get('horizon', 365))
        )
        
        # Создаем будущие признаки
        future_df = pd.DataFrame({'Дата': forecast_dates})
        future_df['day_of_week'] = future_df['Дата'].dt.dayofweek
        future_df['month'] = future_df['Дата'].dt.month
        future_df['day'] = future_df['Дата'].dt.day
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Для прогноза используем последние известные значения для лагов
        last_values = df_features[metric].tail(30).values
        predictions = []
        
        for i in range(len(future_df)):
            # Создаем лаговые признаки из последних предсказаний
            lags = {}
            for lag in [1, 7, 14, 30]:
                if i >= lag:
                    lags[f'lag_{lag}'] = predictions[i-lag]
                else:
                    if len(last_values) >= lag - i:
                        lags[f'lag_{lag}'] = last_values[-(lag-i)]
                    else:
                        lags[f'lag_{lag}'] = np.mean(last_values)
            
            X_pred = pd.DataFrame([{
                **future_df.iloc[i][features[:4]].to_dict(),
                **lags
            }])
            
            pred = model.predict(X_pred)[0]
            predictions.append(pred)
        
        # Объединяем исторические и прогнозные данные
        forecast_df = pd.DataFrame({
            'ds': pd.concat([df['Дата'], future_df['Дата']]),
            'yhat': pd.concat([df[metric], pd.Series(predictions)])
        })
        
        # Добавляем доверительные интервалы
        std_error = np.std(y_train - model.predict(X_train))
        confidence_interval = params.get('confidence_interval', self.forecast_defaults.get('confidence_interval', 0.95))
        z_score = norm.ppf((1 + confidence_interval) / 2)
        
        forecast_df['yhat_lower'] = forecast_df['yhat'] - z_score * std_error
        forecast_df['yhat_upper'] = forecast_df['yhat'] + z_score * std_error
        
        # Для исторических данных интервалы равны значениям
        historical_len = len(df[metric])
        forecast_df.loc[:historical_len-1, 'yhat_lower'] = forecast_df.loc[:historical_len-1, 'yhat']
        forecast_df.loc[:historical_len-1, 'yhat_upper'] = forecast_df.loc[:historical_len-1, 'yhat']
        
        return forecast_df
    
    def forecast_lstm(self, df: pd.DataFrame, metric: str, params: dict) -> pd.DataFrame:
        """Прогнозирование с помощью LSTM"""
        # Нормализация данных
        scaler = StandardScaler()
        data = df[metric].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)
        
        # Получаем параметры из конфига
        lstm_params = self.forecast_defaults.get('lstm', {})
        
        # Параметры
        lookback = params.get('lstm_lookback', lstm_params.get('lookback', 30))
        train_size = int(len(scaled_data) * params.get('train_split', self.forecast_defaults.get('train_split', 0.8)))
        
        # Подготовка данных
        X, y = self.prepare_time_series_data(df[metric].values, lookback)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Разделение на обучающую выборку
        X_train = torch.FloatTensor(X[:train_size])
        y_train = torch.FloatTensor(y[:train_size])
        
        # Создание модели
        model = LSTMModel(
            input_size=1,
            hidden_size=params.get('lstm_hidden_size', lstm_params.get('hidden_size', 50)),
            num_layers=params.get('lstm_num_layers', lstm_params.get('num_layers', 2)),
            output_size=1
        )
        
        # Обучение
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.get('lstm_learning_rate', lstm_params.get('learning_rate', 0.001))
        )
        
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(
            dataset,
            batch_size=params.get('lstm_batch_size', lstm_params.get('batch_size', 32)),
            shuffle=True
        )
        
        epochs = params.get('lstm_epochs', lstm_params.get('epochs', 50))
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
        
        # Прогнозирование
        model.eval()
        with torch.no_grad():
            # Прогноз на будущее
            last_sequence = torch.FloatTensor(scaled_data[-lookback:].reshape(1, lookback, 1))
            predictions = []
            
            for _ in range(params.get('forecast_horizon', self.forecast_defaults.get('horizon', 365))):
                pred = model(last_sequence)
                predictions.append(pred.item())
                
                # Обновляем последовательность
                new_sequence = torch.cat([last_sequence[:, 1:, :], pred.reshape(1, 1, 1)], dim=1)
                last_sequence = new_sequence
        
        # Денормализация
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Создаем DataFrame с результатами
        last_date = df['Дата'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions))
        
        forecast_df = pd.DataFrame({
            'ds': pd.concat([df['Дата'], pd.Series(future_dates)]),
            'yhat': pd.concat([df[metric], pd.Series(predictions)])
        })
        
        # Добавляем доверительные интервалы
        # Расчет остатков на обучающей выборке
        model.eval()
        with torch.no_grad():
            train_predictions = model(X_train).squeeze().numpy()
        residuals = y_train.numpy() - train_predictions
        std_error_train = np.std(residuals)
        
        confidence_interval = params.get('confidence_interval', self.forecast_defaults.get('confidence_interval', 0.95))
        z_score = norm.ppf((1 + confidence_interval) / 2)
        
        # Применяем интервалы только к прогнозной части
        forecast_len = len(predictions)
        hist_len = len(forecast_df) - forecast_len
        
        forecast_df['yhat_lower'] = forecast_df['yhat']
        forecast_df['yhat_upper'] = forecast_df['yhat']
        
        forecast_df.iloc[hist_len:, forecast_df.columns.get_loc('yhat_lower')] = (
            forecast_df.iloc[hist_len:]['yhat'] - z_score * std_error_train
        )
        forecast_df.iloc[hist_len:, forecast_df.columns.get_loc('yhat_upper')] = (
            forecast_df.iloc[hist_len:]['yhat'] + z_score * std_error_train
        )
        
        return forecast_df
    
    def forecast_cafe(self, cafe: str, params: dict, progress_callback=None) -> Optional[pd.DataFrame]:
        """
        Прогнозирование для одного кафе
        
        Args:
            cafe: Название кафе
            params: Параметры прогнозирования
            progress_callback: Функция обратного вызова для отслеживания прогресса
            
        Returns:
            DataFrame с результатами прогноза
        """
        try:
            # Получаем данные кафе
            df_cafe = self.data_loader.get_cafe_data(cafe)
            
            # Получаем настройки для конкретного кафе
            cafe_settings = params.get('cafe_settings', {}).get(cafe, {})
            
            # Создаем локальную копию параметров и обновляем их настройками кафе
            local_params = params.copy()
            if cafe_settings:
                # Обновляем параметры настройками конкретного кафе
                local_params.update(cafe_settings)
            
            # Подготовка данных
            df_cafe = self.data_loader.prepare_for_forecast(
                df_cafe,
                remove_outliers=local_params.get('remove_outliers', True),
                outlier_multiplier=local_params.get('outlier_multiplier', 1.5)
            )
            
            # Выбор модели
            model_type = local_params.get('model_type', 'prophet')
            
            # Автоматическая настройка гиперпараметров
            if local_params.get('auto_tune', False):
                auto_tune_functions = {
                    'prophet': self.auto_tune_prophet,
                    'arima': self.auto_tune_arima,
                    'xgboost': self.auto_tune_xgboost,
                    'lstm': self.auto_tune_lstm
                }
                
                auto_tune_func = auto_tune_functions.get(model_type)
                
                if auto_tune_func:
                    n_trials = self.auto_tuning_config.get('n_trials_quick', 20)
                    if len(df_cafe) >= 365:
                        n_trials = self.auto_tuning_config.get('n_trials_full', 50)
                    
                    if progress_callback:
                        progress_callback(f"Автонастройка параметров для {cafe}, модель {model_type}...")
                    
                    # Автонастройка для трафика
                    traffic_tuning = auto_tune_func(df_cafe, 'Тр', local_params.copy(), n_trials)
                    
                    if progress_callback:
                        progress_callback(f"Лучшая MAPE для трафика: {traffic_tuning['mape']:.2f}%")
                    
                    # Автонастройка для среднего чека
                    check_tuning = auto_tune_func(df_cafe, 'Чек', local_params.copy(), n_trials)
                    
                    if progress_callback:
                        progress_callback(f"Лучшая MAPE для среднего чека: {check_tuning['mape']:.2f}%")
                    
                    # Выбираем лучшие параметры
                    if traffic_tuning['mape'] < check_tuning['mape']:
                        local_params.update(traffic_tuning['params'])
                    else:
                        local_params.update(check_tuning['params'])
            
            # Функции прогнозирования
            forecast_functions = {
                'prophet': self.forecast_prophet,
                'arima': self.forecast_arima,
                'xgboost': self.forecast_xgboost,
                'lstm': self.forecast_lstm
            }
            
            forecast_func = forecast_functions.get(model_type, self.forecast_prophet)
            
            # Прогнозирование трафика
            forecast_traffic = forecast_func(df_cafe, 'Тр', local_params)
            
            # Прогнозирование среднего чека
            forecast_check = forecast_func(df_cafe, 'Чек', local_params)
            
            # Объединение результатов
            result = pd.DataFrame({
                'ds': forecast_traffic['ds'],
                'traffic_forecast': forecast_traffic['yhat'],
                'traffic_lower': forecast_traffic['yhat_lower'],
                'traffic_upper': forecast_traffic['yhat_upper'],
                'check_forecast': forecast_check['yhat'],
                'check_lower': forecast_check['yhat_lower'],
                'check_upper': forecast_check['yhat_upper']
            })
            
            # Расчет прогноза выручки
            result['revenue_forecast'] = result['traffic_forecast'] * result['check_forecast']
            result['revenue_lower'] = result['traffic_lower'] * result['check_lower']
            result['revenue_upper'] = result['traffic_upper'] * result['check_upper']
            
            # Добавление фактических данных
            result = result.merge(
                df_cafe[['Дата', 'Тр', 'Чек', 'Выручка']].rename(columns={
                    'Дата': 'ds',
                    'Тр': 'traffic_fact',
                    'Чек': 'check_fact',
                    'Выручка': 'revenue_fact'
                }),
                on='ds',
                how='left'
            )
            
            result['cafe'] = cafe
            
            # Применение корректировок
            result = self._apply_adjustments(result, cafe, params.get('adjustments', []))
            
            # Расчет метрик качества
            metrics = self._calculate_cafe_metrics(result)
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования для {cafe}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _apply_adjustments(self, result: pd.DataFrame, cafe: str, adjustments: List[dict]) -> pd.DataFrame:
        """Применение ручных корректировок к прогнозу"""
        if not adjustments:
            return result
        
        result = result.copy()
        adjusted_rows_mask = pd.Series(False, index=result.index)
        
        for adj in adjustments:
            if adj['cafe'] == cafe or adj['cafe'] == 'ALL':
                date_from = pd.to_datetime(adj['date_from'])
                date_to = pd.to_datetime(adj['date_to'])
                current_adj_mask = (result['ds'] >= date_from) & (result['ds'] <= date_to)
                
                if adj['metric'] == 'traffic':
                    result.loc[current_adj_mask, 'traffic_forecast'] *= adj['coefficient']
                elif adj['metric'] == 'check':
                    result.loc[current_adj_mask, 'check_forecast'] *= adj['coefficient']
                elif adj['metric'] == 'both':
                    result.loc[current_adj_mask, 'traffic_forecast'] *= adj['coefficient']
                    result.loc[current_adj_mask, 'check_forecast'] *= adj['coefficient']
                
                adjusted_rows_mask |= current_adj_mask
        
        # Пересчитываем выручку для скорректированных строк
        if adjusted_rows_mask.any():
            result.loc[adjusted_rows_mask, 'revenue_forecast'] = (
                result.loc[adjusted_rows_mask, 'traffic_forecast'] * 
                result.loc[adjusted_rows_mask, 'check_forecast']
            )
            result.loc[adjusted_rows_mask, 'revenue_lower'] = (
                result.loc[adjusted_rows_mask, 'traffic_lower'] * 
                result.loc[adjusted_rows_mask, 'check_lower']
            )
            result.loc[adjusted_rows_mask, 'revenue_upper'] = (
                result.loc[adjusted_rows_mask, 'traffic_upper'] * 
                result.loc[adjusted_rows_mask, 'check_upper']
            )
        
        return result
    
    def _calculate_cafe_metrics(self, result: pd.DataFrame) -> dict:
        """Расчет метрик качества для кафе"""
        metrics = {}
        historical_mask = result['revenue_fact'].notna()
        
        if historical_mask.sum() > 0:
            # Метрики для выручки
            revenue_metrics = self.calculate_metrics(
                result.loc[historical_mask, 'revenue_fact'].values,
                result.loc[historical_mask, 'revenue_forecast'].values
            )
            metrics.update({
                'MAPE_revenue': revenue_metrics['MAPE'],
                'MAE_revenue': revenue_metrics['MAE'],
                'RMSE_revenue': revenue_metrics['RMSE'],
                'R2_revenue': revenue_metrics['R2']
            })
            
            # Метрики для трафика
            traffic_mask = result['traffic_fact'].notna() & result['traffic_forecast'].notna()
            if traffic_mask.sum() > 0:
                traffic_metrics = self.calculate_metrics(
                    result.loc[traffic_mask, 'traffic_fact'].values,
                    result.loc[traffic_mask, 'traffic_forecast'].values
                )
                metrics.update({
                    'MAPE_traffic': traffic_metrics['MAPE'],
                    'MAE_traffic': traffic_metrics['MAE'],
                    'RMSE_traffic': traffic_metrics['RMSE'],
                    'R2_traffic': traffic_metrics['R2']
                })
            
            # Метрики для среднего чека
            check_mask = result['check_fact'].notna() & result['check_forecast'].notna()
            if check_mask.sum() > 0:
                check_metrics = self.calculate_metrics(
                    result.loc[check_mask, 'check_fact'].values,
                    result.loc[check_mask, 'check_forecast'].values
                )
                metrics.update({
                    'MAPE_check': check_metrics['MAPE'],
                    'MAE_check': check_metrics['MAE'],
                    'RMSE_check': check_metrics['RMSE'],
                    'R2_check': check_metrics['R2']
                })
        
        return metrics
    
    def forecast_all(self, cafes: List[str], params: dict, progress_callback=None) -> Tuple[pd.DataFrame, dict]:
        """
        Прогнозирование для всех выбранных кафе
        
        Args:
            cafes: Список кафе
            params: Параметры прогнозирования
            progress_callback: Функция обратного вызова для прогресса
            
        Returns:
            DataFrame с прогнозами и словарь с метриками
        """
        results = []
        all_metrics = {}
        total = len(cafes)
        
        for i, cafe in enumerate(cafes):
            if progress_callback:
                progress_callback(i + 1, total, f"Обработка {cafe}")
            
            result, metrics = self.forecast_cafe(cafe, params)
            
            if result is not None:
                results.append(result)
                all_metrics[cafe] = metrics
        
        if results:
            combined = pd.concat(results, ignore_index=True)
            return combined, all_metrics
        else:
            return pd.DataFrame(), {}