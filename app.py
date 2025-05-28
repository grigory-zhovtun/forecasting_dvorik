"""
Веб-приложение для прогнозирования выручки сети кафе "Пироговый Дворик"
Версия 2.0 - с поддержкой множественных моделей
"""

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import json
from datetime import datetime, timedelta
import os
import warnings
import logging
from pathlib import Path
import plotly.graph_objs as go
import plotly.utils
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
import tempfile
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import joblib # For Optuna parallelization
from pickle import PicklingError # For joblib exception handling

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

app = Flask(__name__)

# Глобальные переменные для хранения данных
data_cache = {}
forecast_cache = {}
progress_queue = queue.Queue()
current_task = None
metrics_cache = {}
cancel_flag = threading.Event()

# LSTM модель для временных рядов
class LSTMModel(nn.Module):
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
    def __init__(self):
        self.data_path = r'C:\Users\zhovtun.ga\Documents\automate\check_and_trafic\facts.xlsx'
        self.holidays_path = r'C:\Users\zhovtun.ga\Documents\Notebooks\planing-revenue\holidays_df.xlsx'
        self.load_data()

    def load_data(self):
        """Загрузка данных"""
        try:
            # Загрузка фактических данных
            self.df = pd.read_excel(self.data_path)
            self.df = self.df[~self.df['Кафе'].isin(['ПД-15', 'ПД-20', 'ПД-25', 'Кейтеринг'])]
            self.df = self.df.dropna(subset=['Дата', 'Тр', 'Чек'])
            self.df['Дата'] = pd.to_datetime(self.df['Дата'])

            # Расчет выручки
            self.df['Выручка'] = self.df['Тр'] * self.df['Чек']

            # Загрузка праздников
            self.load_holidays()

            # Сохранение в кэш
            data_cache['df'] = self.df
            data_cache['cafes'] = sorted(self.df['Кафе'].unique().tolist())
            data_cache['date_range'] = {
                'min': self.df['Дата'].min().strftime('%Y-%m-%d'),
                'max': self.df['Дата'].max().strftime('%Y-%m-%d')
            }

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")

    def load_holidays(self):
        """Загрузка праздников"""
            # Сначала попробуем загрузить пользовательские праздники
        try:
            custom_holidays = pd.read_excel(self.holidays_path)
            print(f"Загружено {len(custom_holidays)} пользовательских праздников из {self.holidays_path}")
        except Exception as e:
            print(f"Не удалось загрузить пользовательские праздники: {e}")
            custom_holidays = pd.DataFrame(columns=['ds', 'holiday'])

        # Затем загружаем праздники из нашего предварительно созданного файла
        ru_holidays = pd.DataFrame(columns=['ds', 'holiday'])
        prophet_holidays_file = 'prophet_holidays_2020_2027.csv'
        
        try:
            import os
            if os.path.exists(prophet_holidays_file):
                ru_holidays = pd.read_csv(prophet_holidays_file)
                ru_holidays['ds'] = pd.to_datetime(ru_holidays['ds'])
                print(f"Загружено {len(ru_holidays)} российских праздников из файла {prophet_holidays_file}")
            else:
                print(f"Файл {prophet_holidays_file} не найден, пробуем другие варианты...")
                
                # Пробуем файл с другим именем
                alternative_file = 'prophet_holidays_2020_2026.csv'
                if os.path.exists(alternative_file):
                    ru_holidays = pd.read_csv(alternative_file)
                    ru_holidays['ds'] = pd.to_datetime(ru_holidays['ds'])
                    print(f"Загружено {len(ru_holidays)} российских праздников из файла {alternative_file}")
                else:
                    # В крайнем случае, пробуем использовать встроенную функцию Prophet
                    try:
                        years = list(range(2020, 2027))
                        ru_holidays = make_holidays_df(year_list=years, country='RU')
                        print(f"Создано {len(ru_holidays)} российских праздников с помощью Prophet")
                    except Exception as prophet_error:
                        print(f"Не удалось создать праздники Prophet: {prophet_error}")
                
        except Exception as e2:
            print(f"Ошибка при загрузке праздников: {e2}")

        # Объединяем пользовательские и стандартные праздники
        self.holidays = pd.concat([custom_holidays, ru_holidays]).drop_duplicates().reset_index(drop=True)
        print(f"Всего праздников для использования в Prophet: {len(self.holidays)}")

    def classify_cafe(self, cafe):
        """Классификация кафе"""
        df_cafe = self.df[self.df['Кафе'] == cafe]
        unique_days = df_cafe['Дата'].nunique()

        if unique_days >= 365:
            return "mature"
        elif unique_days >= 180:
            return "established"
        elif unique_days >= 90:
            return "developing"
        else:
            return "new"
            
    def auto_tune_prophet(self, df_cafe, metric, params, n_trials=50):
        """Автоматическая настройка гиперпараметров для Prophet"""
        # Готовим данные
        df_model = df_cafe[['Дата', metric]].rename(columns={'Дата': 'ds', metric: 'y'})
        
        # Доля обучающей выборки
        train_split = params.get('train_split', 0.8)
        train_size = int(len(df_model) * train_split)
        train_df = df_model.iloc[:train_size]
        test_df = df_model.iloc[train_size:]
        
        # Если тестовая выборка слишком маленькая, используем кросс-валидацию
        if len(test_df) < 30:
            test_df = df_model.iloc[-30:]
            
        # Функция цели для оптимизации
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
            
            # Объединяем с фактическими данными
            merged = test_forecast.merge(test_df, on='ds')
            
            # Считаем MAPE
            mape = mean_absolute_percentage_error(merged['y'], merged['yhat']) * 100
            
            return mape
        
        # Запускаем оптимизацию
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, TypeError, PicklingError) as e: # type: ignore # noqa
            print(f"Parallel Optuna execution failed for Prophet: {e}. Falling back to n_jobs=1.")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        # Лучшие параметры
        best_params = study.best_params
        best_mape = study.best_value
        
        # Обновляем параметры
        params.update(best_params)
        
        # Возвращаем лучшие параметры и метрику
        return {'params': best_params, 'mape': best_mape}
        
    def auto_tune_xgboost(self, df_cafe, metric, params, n_trials=50):
        """Автоматическая настройка гиперпараметров для XGBoost"""
        # Подготовка признаков
        df_features = df_cafe.copy()
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
        
        # Доля обучающей выборки
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
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, TypeError, PicklingError) as e: # type: ignore # noqa
            print(f"Parallel Optuna execution failed for XGBoost: {e}. Falling back to n_jobs=1.")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        # Лучшие параметры
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
        
    def auto_tune_arima(self, df_cafe, metric, params, n_trials=30):
        """Автоматическая настройка гиперпараметров для ARIMA"""
        data = df_cafe.set_index('Дата')[metric]
        
        # Доля обучающей выборки
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
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, TypeError, PicklingError) as e: # type: ignore # noqa
            print(f"Parallel Optuna execution failed for ARIMA: {e}. Falling back to n_jobs=1.")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        # Лучшие параметры
        best_params = study.best_params
        best_mape = study.best_value
        
        # Обновляем параметры
        params.update({
            'arima_p': best_params['p'],
            'arima_d': best_params['d'],
            'arima_q': best_params['q']
        })
        
        return {'params': best_params, 'mape': best_mape}
        
    def auto_tune_lstm(self, df_cafe, metric, params, n_trials=20):
        """Автоматическая настройка гиперпараметров для LSTM"""
        # Нормализация данных
        scaler = StandardScaler()
        data = df_cafe[metric].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)
        
        # Параметры
        lookback = params.get('lstm_lookback', 30)
        
        # Подготовка данных
        X, y = self.prepare_time_series_data(df_cafe, metric, lookback)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Доля обучающей выборки
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
            
            # Уменьшаем количество эпох для оптимизации времени
            epochs = 20
            
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
            # Note: LSTM with PyTorch can be tricky with pickling for n_jobs > 1.
            # If issues arise, this might need to be n_jobs=1 or use a different joblib backend.
            study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, TypeError, PicklingError) as e: # type: ignore # noqa
            print(f"Parallel Optuna execution failed for LSTM: {e}. Falling back to n_jobs=1.")
            study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        # Лучшие параметры
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

    def remove_outliers(self, df_cafe, column, multiplier=1.5):
        """Удаление выбросов"""
        Q1 = df_cafe[column].quantile(0.25)
        Q3 = df_cafe[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        return df_cafe[(df_cafe[column] >= lower_bound) & (df_cafe[column] <= upper_bound)]

    def calculate_metrics(self, y_true, y_pred):
        """Расчет метрик качества прогноза"""
        # Удаляем NaN значения для корректного расчета
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {'MAPE': None, 'RMSE': None}

        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return {
            'MAPE': round(mape, 2),
            'RMSE': round(rmse, 2)
        }

    def prepare_time_series_data(self, df_cafe, column, lookback=30):
        """Подготовка данных для моделей временных рядов"""
        data = df_cafe[column].values
        X, y = [], []

        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def forecast_prophet(self, df_cafe, metric, params):
        """Прогнозирование с помощью Prophet"""
        df_model = df_cafe[['Дата', metric]].rename(columns={'Дата': 'ds', metric: 'y'})

        m = Prophet(
            holidays=self.holidays,
            changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
            yearly_seasonality=params.get('yearly_seasonality', True),
            weekly_seasonality=params.get('weekly_seasonality', True),
            daily_seasonality=params.get('daily_seasonality', False),
            interval_width=params.get('confidence_interval', 0.95) # Updated
        )

        m.fit(df_model)

        future = m.make_future_dataframe(periods=params.get('forecast_horizon', 365))
        forecast = m.predict(future)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def forecast_arima(self, df_cafe, metric, params):
        """Прогнозирование с помощью ARIMA"""
        data = df_cafe.set_index('Дата')[metric]

        # Параметры ARIMA (можно сделать настраиваемыми)
        order = (params.get('arima_p', 2), params.get('arima_d', 1), params.get('arima_q', 2))

        model = ARIMA(data, order=order)
        model_fit = model.fit()

        # Прогноз
        forecast_steps = params.get('forecast_horizon', 365)
        forecast = model_fit.forecast(steps=forecast_steps)

        # Создаем даты для прогноза
        last_date = df_cafe['Дата'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps)

        # Создаем DataFrame с прогнозом
        forecast_df = pd.DataFrame({
            'ds': pd.concat([df_cafe['Дата'], pd.Series(future_dates)]),
            'yhat': pd.concat([pd.Series(data.values), forecast])
        })

        # Добавляем доверительные интервалы
        confidence_interval = params.get('confidence_interval', 0.95)
        alpha = 1 - confidence_interval
        
        # Получаем прогнозные значения и стандартные ошибки
        forecast_results = model_fit.get_forecast(steps=forecast_steps)
        forecast_values = forecast_results.predicted_mean
        conf_int = forecast_results.conf_int(alpha=alpha) # ARIMA uses alpha

        # Создаем DataFrame с прогнозом
        forecast_df = pd.DataFrame({
            'ds': pd.concat([df_cafe['Дата'], pd.Series(future_dates)]),
            'yhat': pd.concat([pd.Series(data.values), forecast_values])
        })
        
        # Добавляем границы доверительного интервала
        # Убедимся, что существующие данные не получают интервалы, если они не были рассчитаны
        yhat_lower = pd.Series([np.nan] * len(df_cafe))
        yhat_upper = pd.Series([np.nan] * len(df_cafe))
        
        yhat_lower = pd.concat([yhat_lower, conf_int.iloc[:, 0]]).reset_index(drop=True)
        yhat_upper = pd.concat([yhat_upper, conf_int.iloc[:, 1]]).reset_index(drop=True)

        forecast_df['yhat_lower'] = yhat_lower
        forecast_df['yhat_upper'] = yhat_upper
        
        # Заполняем NaN в yhat для исторических данных оригинальными значениями
        forecast_df['yhat'] = forecast_df['yhat'].fillna(pd.Series(data.values, index=df_cafe.index[:len(data)]))


        return forecast_df

    def forecast_xgboost(self, df_cafe, metric, params):
        """Прогнозирование с помощью XGBoost"""
        # Подготовка признаков
        df_features = df_cafe.copy()
        df_features['day_of_week'] = df_features['Дата'].dt.dayofweek
        df_features['month'] = df_features['Дата'].dt.month
        df_features['day'] = df_features['Дата'].dt.day
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)

        # Создаем лаговые признаки
        for lag in [1, 7, 14, 30]:
            df_features[f'lag_{lag}'] = df_features[metric].shift(lag)

        df_features = df_features.dropna()

        # Разделение на обучающую и тестовую выборки
        train_size = int(len(df_features) * params.get('train_split', 0.8))
        train = df_features[:train_size]

        features = ['day_of_week', 'month', 'day', 'is_weekend'] + [f'lag_{i}' for i in [1, 7, 14, 30]]

        X_train = train[features]
        y_train = train[metric]

        # Обучение модели
        model = xgb.XGBRegressor(
            n_estimators=params.get('xgb_n_estimators', 100),
            max_depth=params.get('xgb_max_depth', 5),
            learning_rate=params.get('xgb_learning_rate', 0.1),
            random_state=42
        )

        model.fit(X_train, y_train)

        # Прогнозирование
        last_date = df_cafe['Дата'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1),
                                     periods=params.get('forecast_horizon', 365))

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
            'ds': pd.concat([df_cafe['Дата'], future_df['Дата']]),
            'yhat': pd.concat([df_cafe[metric], pd.Series(predictions)])
        })

        # Добавляем доверительные интервалы
        std_error = np.std(y_train - model.predict(X_train)) # Оценка стандартной ошибки на обучающей выборке
        confidence_interval = params.get('confidence_interval', 0.95)
        z_score = norm.ppf((1 + confidence_interval) / 2)

        forecast_df['yhat_lower'] = forecast_df['yhat'] - z_score * std_error
        forecast_df['yhat_upper'] = forecast_df['yhat'] + z_score * std_error
        
        # Для исторических данных yhat_lower и yhat_upper могут быть NaN или равны yhat
        historical_len = len(df_cafe[metric])
        forecast_df.loc[:historical_len-1, 'yhat_lower'] = forecast_df.loc[:historical_len-1, 'yhat']
        forecast_df.loc[:historical_len-1, 'yhat_upper'] = forecast_df.loc[:historical_len-1, 'yhat']


        return forecast_df

    def forecast_lstm(self, df_cafe, metric, params):
        """Прогнозирование с помощью LSTM"""
        # Нормализация данных
        scaler = StandardScaler()
        data = df_cafe[metric].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)

        # Параметры
        lookback = params.get('lstm_lookback', 30)
        train_size = int(len(scaled_data) * params.get('train_split', 0.8))

        # Подготовка данных
        X, y = self.prepare_time_series_data(df_cafe, metric, lookback)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Разделение на обучающую и тестовую выборки
        X_train = torch.FloatTensor(X[:train_size])
        y_train = torch.FloatTensor(y[:train_size])

        # Создание модели
        model = LSTMModel(
            input_size=1,
            hidden_size=params.get('lstm_hidden_size', 50),
            num_layers=params.get('lstm_num_layers', 2),
            output_size=1
        )

        # Обучение
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('lstm_learning_rate', 0.001))

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        epochs = params.get('lstm_epochs', 50)
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

            for _ in range(params.get('forecast_horizon', 365)):
                pred = model(last_sequence)
                predictions.append(pred.item())

                # Обновляем последовательность
                new_sequence = torch.cat([last_sequence[:, 1:, :], pred.reshape(1, 1, 1)], dim=1)
                last_sequence = new_sequence

        # Денормализация
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # Создаем DataFrame с результатами
        last_date = df_cafe['Дата'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions))

        forecast_df = pd.DataFrame({
            'ds': pd.concat([df_cafe['Дата'], pd.Series(future_dates)]),
            'yhat': pd.concat([df_cafe[metric], pd.Series(predictions)])
        })

        # Добавляем доверительные интервалы
        # Для LSTM оценка std_error может быть сложной.
        # Используем стандартное отклонение остатков на обучающих данных, если возможно,
        # или более простой эвристический подход.
        # Здесь используется std обучающих данных y_train, что является грубой оценкой.
        # Более точный подход потребовал бы оценки неопределенности модели (например, через dropout MC).
        
        # Расчет остатков на обучающей выборке
        model.eval()
        with torch.no_grad():
            train_predictions = model(X_train).squeeze().numpy()
        residuals = y_train.numpy() - train_predictions
        std_error_train = np.std(residuals) # Используем std остатков на трейне

        confidence_interval = params.get('confidence_interval', 0.95)
        z_score = norm.ppf((1 + confidence_interval) / 2)

        # Применяем интервалы только к прогнозной части
        forecast_len = len(predictions)
        hist_len = len(forecast_df) - forecast_len
        
        forecast_df['yhat_lower'] = forecast_df['yhat'] # по умолчанию
        forecast_df['yhat_upper'] = forecast_df['yhat'] # по умолчанию

        forecast_df.iloc[hist_len:, forecast_df.columns.get_loc('yhat_lower')] = forecast_df.iloc[hist_len:]['yhat'] - z_score * std_error_train
        forecast_df.iloc[hist_len:, forecast_df.columns.get_loc('yhat_upper')] = forecast_df.iloc[hist_len:]['yhat'] + z_score * std_error_train


        return forecast_df

    def forecast_cafe(self, cafe, params):
        """Прогнозирование для одного кафе"""
        try:
            # Фильтрация данных
            df_cafe = self.df[self.df['Кафе'] == cafe].copy()
            df_cafe = df_cafe.groupby('Дата').agg({
                'Тр': 'sum',
                'Чек': 'mean',
                'Выручка': 'sum'
            }).reset_index()

            # Удаление выбросов если включено
            if params.get('remove_outliers', True):
                multiplier = params.get('outlier_multiplier', 1.5)
                df_cafe = self.remove_outliers(df_cafe, 'Тр', multiplier)
                df_cafe = self.remove_outliers(df_cafe, 'Чек', multiplier)

            # Заполнение пропусков
            df_cafe = df_cafe.set_index('Дата').asfreq('D').reset_index()
            df_cafe['Тр'] = df_cafe['Тр'].fillna(df_cafe['Тр'].mean())
            df_cafe['Чек'] = df_cafe['Чек'].fillna(df_cafe['Чек'].mean())
            df_cafe['Выручка'] = df_cafe['Тр'] * df_cafe['Чек']

            # Выбор модели для прогнозирования
            model_type = params.get('model_type', 'prophet')

            # Автоматическая настройка гиперпараметров, если включена
            if params.get('auto_tune', False):
                # Словарь функций автонастройки для разных моделей
                auto_tune_functions = {
                    'prophet': self.auto_tune_prophet,
                    'arima': self.auto_tune_arima,
                    'xgboost': self.auto_tune_xgboost,
                    'lstm': self.auto_tune_lstm
                }
                
                # Выбираем нужную функцию автонастройки
                auto_tune_func = auto_tune_functions.get(model_type)
                
                if auto_tune_func:
                    # Число попыток оптимизации зависит от размера данных
                    n_trials = 20 if len(df_cafe) < 365 else 50
                    
                    # Обновляем параметры в progress_queue, если есть функция обратного вызова
                    progress_message = f"Автонастройка параметров для {cafe}, модель {model_type}..."
                    progress_queue.put({
                        'message': progress_message,
                        'status': 'auto_tuning'
                    })
                    
                    # Запускаем автонастройку для трафика
                    traffic_tuning = auto_tune_func(df_cafe, 'Тр', params.copy(), n_trials)
                    
                    # Обновляем сообщение о прогрессе
                    progress_message = f"Лучшая MAPE для трафика: {traffic_tuning['mape']:.2f}%"
                    progress_queue.put({
                        'message': progress_message,
                        'status': 'auto_tuning'
                    })
                    
                    # Запускаем автонастройку для среднего чека
                    check_tuning = auto_tune_func(df_cafe, 'Чек', params.copy(), n_trials)
                    
                    # Обновляем сообщение о прогрессе
                    progress_message = f"Лучшая MAPE для среднего чека: {check_tuning['mape']:.2f}%"
                    progress_queue.put({
                        'message': progress_message,
                        'status': 'auto_tuning'
                    })
                    
                    # Объединяем лучшие параметры из обеих автонастроек
                    # В случае конфликта приоритет отдаем параметрам с лучшей метрикой
                    if traffic_tuning['mape'] < check_tuning['mape']:
                        params.update(traffic_tuning['params'])
                    else:
                        params.update(check_tuning['params'])
                    
                    # Сохраняем лучшие метрики
                    if 'auto_tune_metrics' not in metrics_cache:
                        metrics_cache['auto_tune_metrics'] = {}
                    
                    metrics_cache['auto_tune_metrics'][cafe] = {
                        'traffic_mape': traffic_tuning['mape'],
                        'check_mape': check_tuning['mape'],
                        'params': params
                    }
    
            # Словарь с функциями прогнозирования
            forecast_functions = {
                'prophet': self.forecast_prophet,
                'arima': self.forecast_arima,
                'xgboost': self.forecast_xgboost,
                'lstm': self.forecast_lstm
            }

            forecast_func = forecast_functions.get(model_type, self.forecast_prophet)

            # Прогнозирование трафика
            forecast_traffic = forecast_func(df_cafe, 'Тр', params)

            # Прогнозирование среднего чека
            forecast_check = forecast_func(df_cafe, 'Чек', params)

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
            if 'adjustments' in params and params['adjustments']:
                adjusted_rows_mask = pd.Series(False, index=result.index) # Keep track of adjusted rows

                for adj in params['adjustments']:
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
                        
                        adjusted_rows_mask |= current_adj_mask # Accumulate masks of adjusted rows

                # Recalculate revenue for rows that were affected by any adjustment
                if adjusted_rows_mask.any():
                    result.loc[adjusted_rows_mask, 'revenue_forecast'] = result.loc[adjusted_rows_mask, 'traffic_forecast'] * result.loc[adjusted_rows_mask, 'check_forecast']
                    # Recalculate revenue bounds as well, using their original components.
                    # Assuming traffic_lower/upper and check_lower/upper are not directly adjusted by coefficient.
                    # If they were, that logic would need to be added above similar to forecasts.
                    result.loc[adjusted_rows_mask, 'revenue_lower'] = result.loc[adjusted_rows_mask, 'traffic_lower'] * result.loc[adjusted_rows_mask, 'check_lower']
                    result.loc[adjusted_rows_mask, 'revenue_upper'] = result.loc[adjusted_rows_mask, 'traffic_upper'] * result.loc[adjusted_rows_mask, 'check_upper']

            # Расчет метрик качества для исторических данных
            historical_mask = result['revenue_fact'].notna()
            cafe_metrics = {}
            
            if historical_mask.sum() > 0:
                # Метрики для выручки
                revenue_metrics = self.calculate_metrics(
                    result.loc[historical_mask, 'revenue_fact'].values,
                    result.loc[historical_mask, 'revenue_forecast'].values
                )
                cafe_metrics['MAPE_revenue'] = revenue_metrics['MAPE']
                cafe_metrics['RMSE_revenue'] = revenue_metrics['RMSE']
                
                # Метрики для трафика
                traffic_mask = result['traffic_fact'].notna() & result['traffic_forecast'].notna()
                if traffic_mask.sum() > 0:
                    traffic_metrics = self.calculate_metrics(
                        result.loc[traffic_mask, 'traffic_fact'].values,
                        result.loc[traffic_mask, 'traffic_forecast'].values
                    )
                    cafe_metrics['MAPE_traffic'] = traffic_metrics['MAPE']
                    cafe_metrics['RMSE_traffic'] = traffic_metrics['RMSE']
                else:
                    cafe_metrics['MAPE_traffic'] = None
                    cafe_metrics['RMSE_traffic'] = None
                
                # Метрики для среднего чека
                check_mask = result['check_fact'].notna() & result['check_forecast'].notna()
                if check_mask.sum() > 0:
                    check_metrics = self.calculate_metrics(
                        result.loc[check_mask, 'check_fact'].values,
                        result.loc[check_mask, 'check_forecast'].values
                    )
                    cafe_metrics['MAPE_check'] = check_metrics['MAPE']
                    cafe_metrics['RMSE_check'] = check_metrics['RMSE']
                else:
                    cafe_metrics['MAPE_check'] = None
                    cafe_metrics['RMSE_check'] = None
                    
                metrics_cache[cafe] = cafe_metrics

            return result

        except Exception as e:
            print(f"Ошибка прогнозирования для {cafe}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def forecast_all(self, cafes, params, progress_callback=None):
        """Прогнозирование для всех выбранных кафе"""
        results = []
        total = len(cafes)

        for i, cafe in enumerate(cafes):
            # Проверка флага отмены
            if cancel_flag.is_set():
                progress_queue.put({'status': 'cancelled'})
                return pd.DataFrame()
                
            if progress_callback:
                progress_callback(i + 1, total, f"Обработка {cafe}")

            result = self.forecast_cafe(cafe, params)
            if result is not None:
                results.append(result)

        if results:
            combined = pd.concat(results, ignore_index=True)
            return combined
        else:
            return pd.DataFrame()

    def prepare_export_data(self, forecast_df):
        """Подготовка данных для экспорта в Excel"""
        # Создаем копию для безопасности
        export_df = forecast_df.copy()

        # Проверяем какие колонки есть в датафрейме
        print("Доступные колонки в forecast_df:")
        print(list(export_df.columns))

        # Переименовываем существующие колонки
        rename_map = {
            'ds': 'Дата',
            'cafe': 'Кафе',
            'traffic_fact': 'Трафик_факт',
            'traffic_forecast': 'Трафик_прогноз',
            'check_fact': 'Средний_чек_факт',
            'check_forecast': 'Средний_чек_прогноз',
            'revenue_fact': 'Выручка_факт',
            'revenue_forecast': 'Выручка_прогноз',
            'revenue_lower': 'Выручка_прогноз_мин',
            'revenue_upper': 'Выручка_прогноз_макс'
        }

        # Переименовываем только существующие колонки
        columns_to_rename = {}
        for old_col, new_col in rename_map.items():
            if old_col in export_df.columns:
                columns_to_rename[old_col] = new_col

        export_df = export_df.rename(columns=columns_to_rename)

        # Список желаемых колонок для экспорта
        desired_columns = [
            'Дата', 'Кафе',
            'Трафик_факт', 'Трафик_прогноз',
            'Средний_чек_факт', 'Средний_чек_прогноз',
            'Выручка_факт', 'Выручка_прогноз',
            'Выручка_прогноз_мин', 'Выручка_прогноз_макс'
        ]

        # Выбираем только существующие колонки
        available_columns = [col for col in desired_columns if col in export_df.columns]

        print(f"Колонки для экспорта: {available_columns}")

        return export_df[available_columns]


# Создание экземпляра движка
engine = ForecastEngine()


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/get_initial_data')
def get_initial_data():
    """Получение начальных данных"""
    return jsonify({
        'cafes': data_cache.get('cafes', []),
        'date_range': data_cache.get('date_range', {}),
        'default_params': {
            'forecast_horizon': 365,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'outlier_multiplier': 1.5,
            'remove_outliers': True,
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'model_type': 'prophet',
            'confidence_interval': 0.95, # Default confidence interval
            'train_split': 0.8,
            'auto_tune': False
        }
    })


@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Запуск прогнозирования"""
    global current_task

    params = request.json
    selected_cafes = params.get('cafes', [])

    if not selected_cafes:
        return jsonify({'error': 'Не выбрано ни одного кафе'}), 400

    # Сбрасываем флаг отмены
    cancel_flag.clear()

    # Очищаем очередь прогресса
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except:
            break

    # Запуск прогнозирования в отдельном потоке
    def run_forecast():
        def progress_callback(current, total, message):
            progress_queue.put({
                'current': current,
                'total': total,
                'message': message,
                'percentage': int((current / total) * 100)
            })

        result = engine.forecast_all(selected_cafes, params, progress_callback)
        if not cancel_flag.is_set():  # Сохраняем результат только если не было отмены
            forecast_cache['latest'] = result
            forecast_cache['params'] = params  # Сохраняем параметры для фильтрации
            progress_queue.put({'status': 'complete'})

    current_task = threading.Thread(target=run_forecast)
    current_task.start()

    return jsonify({'status': 'started'})


@app.route('/api/progress')
def get_progress():
    """Получение прогресса выполнения"""
    try:
        progress = progress_queue.get_nowait()
        return jsonify(progress)
    except queue.Empty:
        return jsonify({'status': 'waiting'})


@app.route('/api/cancel_forecast', methods=['POST'])
def cancel_forecast():
    """Отмена текущего прогнозирования"""
    global current_task
    
    # Устанавливаем флаг отмены
    cancel_flag.set()
    
    # Очищаем очередь прогресса
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except:
            break
    
    # Добавляем сообщение об отмене
    progress_queue.put({'status': 'cancelled'})
    
    return jsonify({'status': 'cancelled'})


@app.route('/api/get_forecast_data')
def get_forecast_data():
    """Получение данных прогноза для визуализации"""
    try:
        if 'latest' not in forecast_cache:
            return jsonify({'error': 'Нет данных прогноза'}), 404

        df = forecast_cache['latest'].copy()

        # Получаем параметры прогноза
        forecast_params = forecast_cache.get('params', {})

        # Проверка наличия данных
        if df.empty:
            return jsonify({
                'data': [],
                'layout': {'title': 'Нет данных для отображения'}
            })

        # Фильтрация по датам
        # Get dates from request arguments
        req_date_from = request.args.get('date_from')
        req_date_to = request.args.get('date_to')

        # Get dates from forecast_params (cached display filters)
        # forecast_params is already defined above as forecast_cache.get('params', {})
        cached_date_from = forecast_params.get('date_filter_from')
        cached_date_to = forecast_params.get('date_filter_to')

        # Determine effective dates: prioritize request args, then cached, then default to current month
        effective_date_from = req_date_from if req_date_from else cached_date_from
        effective_date_to = req_date_to if req_date_to else cached_date_to

        if not effective_date_from and not effective_date_to:
            # If no dates from request or cache, default to current month
            today = datetime.now()
            effective_date_from = today.replace(day=1).strftime('%Y-%m-%d')
            
            # Calculate last day of current month
            # Move to a day near the end of the month (e.g., 28th) to safely add days
            # then go to the 1st of next month, then subtract one day.
            next_month_start = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
            last_day_of_current_month = next_month_start - timedelta(days=1)
            effective_date_to = last_day_of_current_month.strftime('%Y-%m-%d')
            
            # Note: We are not updating forecast_cache['params'] with these defaults here,
            # as the cache should ideally reflect explicit user choices or forecast run params.

        # Apply filters using the determined effective dates
        if effective_date_from:
            df = df[df['ds'] >= pd.to_datetime(effective_date_from)]
        if effective_date_to:
            df = df[df['ds'] <= pd.to_datetime(effective_date_to)]

        # Преобразование в формат для Plotly
        traces = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for idx, cafe in enumerate(df['cafe'].unique()):
            cafe_data = df[df['cafe'] == cafe].sort_values('ds')
            color = colors[idx % len(colors)]

            # Факт - только там где есть данные
            fact_data = cafe_data[cafe_data['revenue_fact'].notna()]
            if not fact_data.empty:
                traces.append({
                    'x': fact_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'y': fact_data['revenue_fact'].round(2).tolist(),
                    'name': f'{cafe} - Факт',
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {
                        'width': 2.5,
                        'color': color
                    },
                    'hovertemplate': '<b>%{fullData.name}</b><br>' +
                                     'Дата: %{x}<br>' +
                                     'Выручка: %{y:,.0f} ₽<br>' +
                                     '<extra></extra>'
                })

            # Прогноз
            forecast_data = cafe_data[cafe_data['revenue_forecast'].notna()]
            if not forecast_data.empty:
                traces.append({
                    'x': forecast_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'y': forecast_data['revenue_forecast'].round(2).tolist(),
                    'name': f'{cafe} - Прогноз',
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {
                        'width': 2.5,
                        'dash': 'dash',
                        'color': color
                    },
                    'hovertemplate': '<b>%{fullData.name}</b><br>' +
                                     'Дата: %{x}<br>' +
                                     'Прогноз: %{y:,.0f} ₽<br>' +
                                     '<extra></extra>'
                })

                # Доверительный интервал
                x_values = forecast_data['ds'].dt.strftime('%Y-%m-%d').tolist()
                y_upper = forecast_data['revenue_upper'].round(2).tolist()
                y_lower = forecast_data['revenue_lower'].round(2).tolist()

                # Создаем замкнутый полигон для заливки
                x_rev = x_values[::-1]
                y_lower_rev = y_lower[::-1]

                traces.append({
                    'x': x_values + x_rev,
                    'y': y_upper + y_lower_rev,
                    'fill': 'toself',
                    'fillcolor': f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    'line': {'color': 'transparent'},
                    'showlegend': False,
                    'name': f'{cafe} - Интервал',
                    'type': 'scatter',
                    'hoverinfo': 'skip'
                })

        # Компоновка графика
        layout = {
            'title': {
                'text': 'Прогноз выручки сети "Пироговый Дворик"',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            'xaxis': {
                'title': 'Дата',
                'showgrid': True,
                'gridcolor': 'rgba(128, 128, 128, 0.2)',
                'showline': True,
                'linewidth': 1,
                'linecolor': 'rgba(128, 128, 128, 0.4)'
            },
            'yaxis': {
                'title': 'Выручка, ₽',
                'tickformat': ',.0f',
                'showgrid': True,
                'gridcolor': 'rgba(128, 128, 128, 0.2)',
                'showline': True,
                'linewidth': 1,
                'linecolor': 'rgba(128, 128, 128, 0.4)'
            },
            'hovermode': 'x unified',
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5,
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': 'rgba(128, 128, 128, 0.2)',
                'borderwidth': 1
            },
            'margin': {'l': 80, 'r': 50, 't': 80, 'b': 100},
            'plot_bgcolor': 'white',
            'paper_bgcolor': '#f8f9fa',
            'font': {
                'family': 'Arial, sans-serif',
                'size': 12
            }
        }

        # Добавляем метрики качества
        metrics = {}
        for cafe in df['cafe'].unique():
            if cafe in metrics_cache:
                metrics[cafe] = metrics_cache[cafe]

        # Добавляем метрики автонастройки, если они есть
        auto_tune_metrics = {}
        if 'auto_tune_metrics' in metrics_cache:
            for cafe in df['cafe'].unique():
                if cafe in metrics_cache['auto_tune_metrics']:
                    auto_tune_metrics[cafe] = {
                        'traffic_mape': round(metrics_cache['auto_tune_metrics'][cafe]['traffic_mape'], 2),
                        'check_mape': round(metrics_cache['auto_tune_metrics'][cafe]['check_mape'], 2)
                    }
        
        # Расчет сводных данных для таблицы
        summary_data = {
            'cafes': {},
            'total': {
                'revenue': {'actual': 0, 'forecast': 0, 'total': 0},
                'traffic': {'actual': 0, 'forecast': 0, 'total': 0}
            }
        }

        if not df.empty:
            today = pd.to_datetime(datetime.now().date()) # Дата без времени для корректного сравнения
            
            current_period_df = df.copy() # df уже отфильтрован по date_from и date_to
            
            # Рассчитываем данные для каждого кафе
            for cafe in df['cafe'].unique():
                cafe_df = current_period_df[current_period_df['cafe'] == cafe]
                
                # Актуальные данные (прошлое и сегодня в выбранном диапазоне)
                actual_df = cafe_df[cafe_df['ds'] <= today]
                revenue_actual = actual_df['revenue_fact'].fillna(0).sum()
                traffic_actual = actual_df['traffic_fact'].fillna(0).sum()
                
                # Прогнозные данные (будущее в выбранном диапазоне)
                forecast_df_period = cafe_df[cafe_df['ds'] > today]
                revenue_forecast = forecast_df_period['revenue_forecast'].fillna(0).sum()
                traffic_forecast = forecast_df_period['traffic_forecast'].fillna(0).sum()
                
                # Общие данные
                revenue_total = revenue_actual + revenue_forecast
                traffic_total = traffic_actual + traffic_forecast
                
                summary_data['cafes'][cafe] = {
                    'revenue': {
                        'actual': revenue_actual,
                        'forecast': revenue_forecast,
                        'total': revenue_total
                    },
                    'traffic': {
                        'actual': traffic_actual,
                        'forecast': traffic_forecast,
                        'total': traffic_total
                    }
                }
                
                # Добавляем к общим итогам
                summary_data['total']['revenue']['actual'] += revenue_actual
                summary_data['total']['revenue']['forecast'] += revenue_forecast
                summary_data['total']['revenue']['total'] += revenue_total
                summary_data['total']['traffic']['actual'] += traffic_actual
                summary_data['total']['traffic']['forecast'] += traffic_forecast
                summary_data['total']['traffic']['total'] += traffic_total


        return jsonify({
            'data': traces,
            'layout': layout,
            'metrics': metrics,
            'auto_tune_metrics': auto_tune_metrics,
            'params': forecast_params,
            'summary_data': summary_data
        })

    except Exception as e:
        app.logger.error(f"Ошибка при подготовке данных для графика: {str(e)}")
        return jsonify({
            'error': f'Ошибка при подготовке данных: {str(e)}'
        }), 500


@app.route('/api/export_excel')
def export_excel():
    """Экспорт результатов в Excel"""
    try:
        if 'latest' not in forecast_cache:
            return jsonify({'error': 'Нет данных для экспорта'}), 404

        df = forecast_cache['latest']

        # Отладочная информация
        print(f"Тип данных в кэше: {type(df)}")
        print(f"Размер данных: {df.shape if hasattr(df, 'shape') else 'N/A'}")

        export_df = engine.prepare_export_data(df)

        # Создание временного файла с уникальным именем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f'forecast_{timestamp}.xlsx')

        # Сохранение в Excel
        with pd.ExcelWriter(temp_file, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Прогноз', index=False)

            # Получаем объекты для форматирования
            workbook = writer.book
            worksheet = writer.sheets['Прогноз']

            # Форматы
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3D3D3',
                'border': 1
            })

            date_format = workbook.add_format({'num_format': 'dd.mm.yyyy'})
            num_format = workbook.add_format({'num_format': '#,##0'})
            money_format = workbook.add_format({'num_format': '#,##0.00 ₽'})

            # Применяем форматы к заголовкам
            for col_num, value in enumerate(export_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Устанавливаем ширину колонок и форматы
            for idx, col in enumerate(export_df.columns):
                if col == 'Дата':
                    worksheet.set_column(idx, idx, 12, date_format)
                elif col == 'Кафе':
                    worksheet.set_column(idx, idx, 15)
                elif 'Трафик' in col:
                    worksheet.set_column(idx, idx, 15, num_format)
                else:  # Денежные колонки
                    worksheet.set_column(idx, idx, 18, money_format)

        # Отправка файла
        response = send_file(
            temp_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'forecast_{timestamp}.xlsx'
        )

        # Удаление временного файла после отправки
        def remove_file(response):
            try:
                os.remove(temp_file)
            except:
                pass
            return response

        response.call_on_close(lambda: remove_file(response))

        return response

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        app.logger.error(f"Ошибка при экспорте: {str(e)}\n{error_details}")
        return jsonify({
            'error': f'Ошибка при экспорте: {str(e)}',
            'details': error_details
        }), 500


if __name__ == '__main__':
    # Создание папки для шаблонов если не существует
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    app.run(debug=True, port=5000)