"""
Модуль для генерации рекомендаций по улучшению качества прогноза
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Движок для генерации рекомендаций"""
    
    def __init__(self, config: dict):
        """
        Инициализация движка рекомендаций
        
        Args:
            config: Конфигурация приложения
        """
        self.config = config
        self.thresholds = config.get('metrics_thresholds', {})
        
        # Пороги MAPE по умолчанию
        self.mape_thresholds = self.thresholds.get('mape', {
            'excellent': 5,
            'good': 10,
            'warning': 20,
            'poor': 30
        })
        
        # Пороги RMSE по умолчанию
        self.rmse_thresholds = self.thresholds.get('rmse', {
            'traffic': 100,
            'check': 50,
            'revenue': 10000
        })
    
    def get_recommendations_for_cafe(self, cafe: str, metrics: dict, model_type: str) -> Dict[str, Any]:
        """
        Получение рекомендаций для конкретного кафе
        
        Args:
            cafe: Название кафе
            metrics: Метрики качества прогноза
            model_type: Тип используемой модели
            
        Returns:
            Словарь с рекомендациями
        """
        recommendations = {
            'cafe': cafe,
            'model_type': model_type,
            'metrics_summary': self._get_metrics_summary(metrics),
            'general_recommendations': [],
            'parameter_recommendations': {},
            'model_recommendations': []
        }
        
        # Анализ метрик и генерация рекомендаций
        for metric_type in ['revenue', 'traffic', 'check']:
            mape_key = f'MAPE_{metric_type}'
            rmse_key = f'RMSE_{metric_type}'
            
            if mape_key in metrics and metrics[mape_key] is not None:
                mape_value = metrics[mape_key]
                quality = self._get_mape_quality(mape_value)
                
                if quality in ['warning', 'poor', 'critical']:
                    # Генерируем рекомендации для плохих метрик
                    recs = self._generate_metric_recommendations(
                        metric_type, mape_value, metrics.get(rmse_key), model_type
                    )
                    recommendations['general_recommendations'].extend(recs['general'])
                    recommendations['parameter_recommendations'].update(recs['parameters'])
                    recommendations['model_recommendations'].extend(recs['models'])
        
        # Удаляем дубликаты
        recommendations['general_recommendations'] = list(set(recommendations['general_recommendations']))
        recommendations['model_recommendations'] = list(set(recommendations['model_recommendations']))
        
        # Приоритизация рекомендаций
        recommendations = self._prioritize_recommendations(recommendations)
        
        return recommendations
    
    def _get_metrics_summary(self, metrics: dict) -> dict:
        """Получение сводки по метрикам"""
        summary = {}
        
        for metric_type in ['revenue', 'traffic', 'check']:
            mape_key = f'MAPE_{metric_type}'
            if mape_key in metrics and metrics[mape_key] is not None:
                mape_value = metrics[mape_key]
                quality = self._get_mape_quality(mape_value)
                summary[metric_type] = {
                    'mape': mape_value,
                    'quality': quality,
                    'quality_rus': self._get_quality_rus(quality)
                }
        
        return summary
    
    def _get_mape_quality(self, mape: float) -> str:
        """Определение качества прогноза по MAPE"""
        if mape <= self.mape_thresholds['excellent']:
            return 'excellent'
        elif mape <= self.mape_thresholds['good']:
            return 'good'
        elif mape <= self.mape_thresholds['warning']:
            return 'warning'
        elif mape <= self.mape_thresholds['poor']:
            return 'poor'
        else:
            return 'critical'
    
    def _get_quality_rus(self, quality: str) -> str:
        """Перевод качества на русский"""
        translations = {
            'excellent': 'Отличное',
            'good': 'Хорошее',
            'warning': 'Среднее',
            'poor': 'Плохое',
            'critical': 'Критическое'
        }
        return translations.get(quality, quality)
    
    def _generate_metric_recommendations(self, metric_type: str, mape: float, 
                                       rmse: Optional[float], model_type: str) -> dict:
        """
        Генерация рекомендаций для конкретной метрики
        
        Args:
            metric_type: Тип метрики (revenue, traffic, check)
            mape: Значение MAPE
            rmse: Значение RMSE
            model_type: Тип модели
            
        Returns:
            Словарь с рекомендациями
        """
        recommendations = {
            'general': [],
            'parameters': {},
            'models': []
        }
        
        metric_rus = {
            'revenue': 'выручки',
            'traffic': 'трафика',
            'check': 'среднего чека'
        }[metric_type]
        
        # Общие рекомендации
        if mape > 30:
            recommendations['general'].append(
                f"Критическая ошибка прогноза {metric_rus} (MAPE={mape:.1f}%). "
                "Рекомендуется проверить качество исходных данных."
            )
            recommendations['general'].append(
                "Возможно, в данных есть аномалии или структурные изменения."
            )
        elif mape > 20:
            recommendations['general'].append(
                f"Высокая ошибка прогноза {metric_rus} (MAPE={mape:.1f}%). "
                "Рекомендуется настроить параметры модели."
            )
        
        # Рекомендации по параметрам для разных моделей
        if model_type == 'prophet':
            recommendations['parameters'] = self._get_prophet_recommendations(metric_type, mape)
        elif model_type == 'arima':
            recommendations['parameters'] = self._get_arima_recommendations(metric_type, mape)
        elif model_type == 'xgboost':
            recommendations['parameters'] = self._get_xgboost_recommendations(metric_type, mape)
        elif model_type == 'lstm':
            recommendations['parameters'] = self._get_lstm_recommendations(metric_type, mape)
        
        # Рекомендации по выбору модели
        if mape > 25:
            if model_type != 'xgboost':
                recommendations['models'].append(
                    "Попробуйте XGBoost - эта модель часто показывает лучшие результаты на сложных данных"
                )
            if model_type != 'lstm' and metric_type in ['traffic', 'revenue']:
                recommendations['models'].append(
                    "LSTM может лучше улавливать долгосрочные зависимости в данных"
                )
        
        # Специфичные рекомендации для типов метрик
        if metric_type == 'traffic' and mape > 20:
            recommendations['general'].append(
                "Трафик сильно зависит от дня недели и праздников. "
                "Убедитесь, что сезонность правильно настроена."
            )
        elif metric_type == 'check' and mape > 15:
            recommendations['general'].append(
                "Средний чек может зависеть от промо-акций и изменений в меню. "
                "Используйте корректировки для учета известных событий."
            )
        
        return recommendations
    
    def _get_prophet_recommendations(self, metric_type: str, mape: float) -> dict:
        """Рекомендации для Prophet"""
        params = {}
        
        if mape > 25:
            params['changepoint_prior_scale'] = {
                'current': 0.05,
                'recommended': 0.1,
                'description': 'Увеличьте гибкость тренда для лучшей адаптации к изменениям'
            }
            params['seasonality_prior_scale'] = {
                'current': 10.0,
                'recommended': 5.0,
                'description': 'Уменьшите силу сезонности для снижения переобучения'
            }
        elif mape > 15:
            params['changepoint_prior_scale'] = {
                'current': 0.05,
                'recommended': 0.08,
                'description': 'Немного увеличьте гибкость тренда'
            }
        
        if metric_type == 'traffic':
            params['weekly_seasonality'] = {
                'current': True,
                'recommended': True,
                'description': 'Убедитесь, что недельная сезонность включена для трафика'
            }
            params['holidays_prior_scale'] = {
                'current': 10.0,
                'recommended': 15.0,
                'description': 'Увеличьте влияние праздников на трафик'
            }
        
        return params
    
    def _get_arima_recommendations(self, metric_type: str, mape: float) -> dict:
        """Рекомендации для ARIMA"""
        params = {}
        
        if mape > 20:
            params['order'] = {
                'current': '(2,1,2)',
                'recommended': '(3,1,3) или (5,1,5)',
                'description': 'Попробуйте увеличить порядок модели для учета более сложных паттернов'
            }
            params['seasonal_order'] = {
                'current': None,
                'recommended': '(1,1,1,7)',
                'description': 'Добавьте сезонную компоненту для учета недельных паттернов'
            }
        
        return params
    
    def _get_xgboost_recommendations(self, metric_type: str, mape: float) -> dict:
        """Рекомендации для XGBoost"""
        params = {}
        
        if mape > 20:
            params['n_estimators'] = {
                'current': 100,
                'recommended': 300,
                'description': 'Увеличьте количество деревьев для лучшего обучения'
            }
            params['max_depth'] = {
                'current': 5,
                'recommended': 7,
                'description': 'Увеличьте глубину деревьев для учета сложных зависимостей'
            }
            params['feature_engineering'] = {
                'current': 'basic',
                'recommended': 'advanced',
                'description': 'Добавьте больше признаков: скользящие средние, лаги разных периодов'
            }
        
        return params
    
    def _get_lstm_recommendations(self, metric_type: str, mape: float) -> dict:
        """Рекомендации для LSTM"""
        params = {}
        
        if mape > 25:
            params['hidden_size'] = {
                'current': 50,
                'recommended': 100,
                'description': 'Увеличьте размер скрытого слоя для большей емкости модели'
            }
            params['num_layers'] = {
                'current': 2,
                'recommended': 3,
                'description': 'Добавьте дополнительный слой LSTM'
            }
            params['lookback'] = {
                'current': 30,
                'recommended': 60,
                'description': 'Увеличьте окно просмотра для учета долгосрочных зависимостей'
            }
        elif mape > 15:
            params['epochs'] = {
                'current': 50,
                'recommended': 100,
                'description': 'Увеличьте количество эпох обучения'
            }
            params['dropout'] = {
                'current': 0,
                'recommended': 0.2,
                'description': 'Добавьте dropout для регуляризации'
            }
        
        return params
    
    def _prioritize_recommendations(self, recommendations: dict) -> dict:
        """Приоритизация рекомендаций"""
        # Сортируем общие рекомендации по важности
        priority_keywords = ['критическ', 'аномал', 'качеств', 'высок', 'провер']
        
        def get_priority(rec: str) -> int:
            for i, keyword in enumerate(priority_keywords):
                if keyword in rec.lower():
                    return i
            return len(priority_keywords)
        
        recommendations['general_recommendations'].sort(key=get_priority)
        
        # Ограничиваем количество рекомендаций
        max_general = 5
        max_models = 3
        
        recommendations['general_recommendations'] = recommendations['general_recommendations'][:max_general]
        recommendations['model_recommendations'] = recommendations['model_recommendations'][:max_models]
        
        return recommendations
    
    def get_metric_color_class(self, mape_value: float) -> str:
        """
        Получение CSS класса для цветовой индикации метрики
        
        Args:
            mape_value: Значение MAPE
            
        Returns:
            CSS класс
        """
        quality = self._get_mape_quality(mape_value)
        
        color_map = {
            'excellent': 'metric-excellent',
            'good': 'metric-good',
            'warning': 'metric-warning',
            'poor': 'metric-poor',
            'critical': 'metric-critical'
        }
        
        return color_map.get(quality, 'metric-warning')
    
    def format_recommendation_tooltip(self, recommendation: dict) -> str:
        """
        Форматирование рекомендации для tooltip
        
        Args:
            recommendation: Словарь с рекомендацией
            
        Returns:
            Отформатированная строка
        """
        if 'parameters' in recommendation and recommendation['parameters']:
            params_text = []
            for param, info in recommendation['parameters'].items():
                if isinstance(info, dict) and 'description' in info:
                    params_text.append(f"• {info['description']}")
            
            if params_text:
                return "Рекомендации по параметрам:\\n" + "\\n".join(params_text)
        
        if 'general_recommendations' in recommendation and recommendation['general_recommendations']:
            return "\\n".join(recommendation['general_recommendations'][:2])
        
        return "Нет специфических рекомендаций"