"""
Тесты для приложения прогнозирования выручки
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import yaml

# Импорт модулей приложения
from data_loader import DataLoader
from forecast_engine import ForecastEngine
from presets import PresetManager
from src.utils.recommendations import RecommendationEngine


@pytest.fixture
def test_config():
    """Фикстура с тестовой конфигурацией"""
    return {
        'data': {
            'facts_file': 'test_facts.xlsx',
            'holidays_file': 'test_holidays.xlsx',
            'prophet_holidays_file': 'prophet_holidays_2020_2027.csv'
        },
        'database': {
            'path': ':memory:'  # SQLite в памяти для тестов
        },
        'excluded_cafes': ['ПД-15', 'ПД-20'],
        'metrics_thresholds': {
            'mape': {
                'excellent': 5,
                'good': 10,
                'warning': 20,
                'poor': 30
            }
        },
        'forecast_defaults': {
            'horizon': 30,
            'confidence_interval': 0.95,
            'train_split': 0.8
        }
    }


@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    data = []
    cafes = ['ПД-01', 'ПД-02', 'ПД-03']
    
    for cafe in cafes:
        for date in dates:
            # Добавляем сезонность и случайный шум
            base_traffic = 100 + 50 * np.sin(2 * np.pi * date.dayofyear / 365)
            traffic = base_traffic + np.random.normal(0, 10)
            
            base_check = 500 + 100 * np.cos(2 * np.pi * date.dayofweek / 7)
            check = base_check + np.random.normal(0, 20)
            
            data.append({
                'Дата': date,
                'Кафе': cafe,
                'Тр': max(0, traffic),
                'Чек': max(0, check)
            })
    
    return pd.DataFrame(data)


class TestDataLoader:
    """Тесты для DataLoader"""
    
    def test_data_loader_init(self, test_config):
        """Тест инициализации DataLoader"""
        loader = DataLoader(test_config)
        assert loader.config == test_config
        assert loader.excluded_cafes == test_config['excluded_cafes']
    
    def test_remove_outliers(self, test_config, sample_data):
        """Тест удаления выбросов"""
        loader = DataLoader(test_config)
        
        # Добавляем выбросы
        df = sample_data.copy()
        df.loc[10, 'Тр'] = 1000  # Выброс
        
        # Удаляем выбросы
        df_cleaned = loader.remove_outliers(df, 'Тр', multiplier=1.5)
        
        # Проверяем, что выброс удален
        assert len(df_cleaned) < len(df)
        assert df_cleaned['Тр'].max() < 1000
    
    def test_prepare_for_forecast(self, test_config, sample_data):
        """Тест подготовки данных для прогнозирования"""
        loader = DataLoader(test_config)
        
        cafe_data = sample_data[sample_data['Кафе'] == 'ПД-01']
        prepared = loader.prepare_for_forecast(cafe_data, remove_outliers=True)
        
        # Проверяем, что данные подготовлены
        assert 'Выручка' in prepared.columns
        assert prepared['Выручка'].notna().all()
        assert len(prepared) == 365  # Все дни года
    
    def test_classify_cafe(self, test_config, sample_data):
        """Тест классификации кафе"""
        loader = DataLoader(test_config)
        loader._cache['facts'] = sample_data
        
        # Кафе с полным годом данных
        classification = loader.classify_cafe('ПД-01')
        assert classification == 'mature'


class TestForecastEngine:
    """Тесты для ForecastEngine"""
    
    def test_forecast_engine_init(self, test_config):
        """Тест инициализации ForecastEngine"""
        loader = DataLoader(test_config)
        engine = ForecastEngine(test_config, loader)
        
        assert engine.config == test_config
        assert engine.data_loader == loader
    
    def test_calculate_metrics(self, test_config):
        """Тест расчета метрик"""
        loader = DataLoader(test_config)
        engine = ForecastEngine(test_config, loader)
        
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([110, 190, 310, 390])
        
        metrics = engine.calculate_metrics(y_true, y_pred)
        
        assert 'MAPE' in metrics
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        assert metrics['MAPE'] > 0
        assert metrics['R2'] > 0.9  # Хорошее соответствие
    
    def test_prepare_time_series_data(self, test_config):
        """Тест подготовки данных для временных рядов"""
        loader = DataLoader(test_config)
        engine = ForecastEngine(test_config, loader)
        
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X, y = engine.prepare_time_series_data(data, lookback=3)
        
        assert len(X) == len(y)
        assert len(X) == 7  # 10 - 3
        assert X[0].tolist() == [1, 2, 3]
        assert y[0] == 4


class TestRecommendationEngine:
    """Тесты для RecommendationEngine"""
    
    def test_recommendation_engine_init(self, test_config):
        """Тест инициализации RecommendationEngine"""
        engine = RecommendationEngine(test_config)
        
        assert engine.config == test_config
        assert engine.mape_thresholds == test_config['metrics_thresholds']['mape']
    
    def test_get_mape_quality(self, test_config):
        """Тест определения качества по MAPE"""
        engine = RecommendationEngine(test_config)
        
        assert engine._get_mape_quality(3) == 'excellent'
        assert engine._get_mape_quality(8) == 'good'
        assert engine._get_mape_quality(15) == 'warning'
        assert engine._get_mape_quality(25) == 'poor'
        assert engine._get_mape_quality(35) == 'critical'
    
    def test_get_recommendations_for_cafe(self, test_config):
        """Тест генерации рекомендаций"""
        engine = RecommendationEngine(test_config)
        
        metrics = {
            'MAPE_revenue': 25,
            'MAPE_traffic': 30,
            'MAPE_check': 15,
            'RMSE_revenue': 10000,
            'MAE_revenue': 5000
        }
        
        recommendations = engine.get_recommendations_for_cafe('ПД-01', metrics, 'prophet')
        
        assert 'general_recommendations' in recommendations
        assert 'parameter_recommendations' in recommendations
        assert 'model_recommendations' in recommendations
        assert len(recommendations['general_recommendations']) > 0


class TestPresetManager:
    """Тесты для PresetManager"""
    
    def test_preset_manager_init(self, test_config):
        """Тест инициализации PresetManager"""
        manager = PresetManager(test_config)
        
        assert manager.config == test_config
        assert manager.db_path == ':memory:'
    
    def test_save_and_load_preset(self, test_config):
        """Тест сохранения и загрузки пресета"""
        manager = PresetManager(test_config)
        
        # Сохраняем пресет
        cafes = ['ПД-01', 'ПД-02']
        params = {'forecast_horizon': 30, 'model_type': 'prophet'}
        adjustments = [{'cafe': 'ПД-01', 'coefficient': 1.1}]
        
        preset_id = manager.save_preset('Тестовый пресет', cafes, params, adjustments)
        assert preset_id > 0
        
        # Загружаем пресет
        preset = manager.load_preset(preset_id)
        assert preset is not None
        assert preset['name'] == 'Тестовый пресет'
        assert preset['cafes'] == cafes
        assert preset['params'] == params
        assert preset['adjustments'] == adjustments
    
    def test_list_presets(self, test_config):
        """Тест получения списка пресетов"""
        manager = PresetManager(test_config)
        
        # Сохраняем несколько пресетов
        manager.save_preset('Пресет 1', ['ПД-01'], {}, [])
        manager.save_preset('Пресет 2', ['ПД-02'], {}, [])
        
        # Получаем список
        presets = manager.list_presets()
        assert len(presets) == 2
        assert presets[0]['name'] in ['Пресет 1', 'Пресет 2']
    
    def test_delete_preset(self, test_config):
        """Тест удаления пресета"""
        manager = PresetManager(test_config)
        
        # Сохраняем пресет
        preset_id = manager.save_preset('Удаляемый пресет', ['ПД-01'], {}, [])
        
        # Удаляем
        success = manager.delete_preset(preset_id)
        assert success
        
        # Проверяем, что пресет удален
        preset = manager.load_preset(preset_id)
        assert preset is None
    
    def test_save_and_list_adjustments(self, test_config):
        """Тест сохранения и получения корректировок"""
        manager = PresetManager(test_config)
        
        # Сохраняем корректировку
        adj_id = manager.save_adjustment(
            cafe='ПД-01',
            date_from='2024-01-01',
            date_to='2024-01-31',
            metric='traffic',
            coefficient=1.2,
            reason='Новогодние праздники'
        )
        assert adj_id > 0
        
        # Получаем список
        adjustments = manager.list_adjustments()
        assert len(adjustments) == 1
        assert adjustments[0]['cafe'] == 'ПД-01'
        assert adjustments[0]['coefficient'] == 1.2


class TestIntegration:
    """Интеграционные тесты"""
    
    @pytest.mark.slow
    def test_full_forecast_workflow(self, test_config, sample_data):
        """Тест полного цикла прогнозирования"""
        # Создаем временный файл с данными
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            sample_data.to_excel(tmp.name, index=False)
            test_config['data']['facts_file'] = tmp.name
        
        try:
            # Инициализация компонентов
            loader = DataLoader(test_config)
            loader._cache['facts'] = sample_data
            engine = ForecastEngine(test_config, loader)
            
            # Прогнозирование
            params = {
                'forecast_horizon': 30,
                'model_type': 'prophet',
                'remove_outliers': True,
                'outlier_multiplier': 1.5,
                'confidence_interval': 0.95,
                'train_split': 0.8
            }
            
            result, metrics = engine.forecast_cafe('ПД-01', params)
            
            # Проверки
            assert result is not None
            assert not result.empty
            assert 'revenue_forecast' in result.columns
            assert metrics is not None
            assert 'MAPE_revenue' in metrics
            
        finally:
            # Удаляем временный файл
            os.unlink(test_config['data']['facts_file'])


# Запуск тестов
if __name__ == '__main__':
    pytest.main([__file__, '-v'])