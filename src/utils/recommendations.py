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
        
        # Расширенные пороги MAPE по умолчанию с детальными рангами
        self.mape_thresholds = self.thresholds.get('mape', {
            'excellent': 5,      # Отличный прогноз
            'good': 10,         # Хороший прогноз
            'acceptable': 15,   # Приемлемый прогноз
            'warning': 20,      # Требует внимания
            'poor': 30,         # Плохой прогноз
            'critical': 40      # Критический прогноз
        })
        
        # Детальные рекомендации по рангам
        self.rank_recommendations = {
            'excellent': {
                'general': 'Отличное качество прогноза. Модель хорошо настроена.',
                'action': 'Продолжайте использовать текущие настройки.',
                'tips': ['Периодически проверяйте актуальность модели', 'Отслеживайте изменения в бизнес-процессах']
            },
            'good': {
                'general': 'Хорошее качество прогноза. Возможны небольшие улучшения.',
                'action': 'Рассмотрите тонкую настройку параметров.',
                'tips': ['Проверьте сезонные компоненты', 'Убедитесь в учете всех праздников']
            },
            'acceptable': {
                'general': 'Приемлемое качество прогноза. Есть потенциал для улучшения.',
                'action': 'Рекомендуется оптимизация модели.',
                'tips': ['Увеличьте объем обучающей выборки', 'Проверьте наличие выбросов в данных']
            },
            'warning': {
                'general': 'Качество прогноза требует внимания.',
                'action': 'Необходима настройка параметров модели.',
                'tips': ['Проанализируйте аномалии в данных', 'Попробуйте другую модель прогнозирования']
            },
            'poor': {
                'general': 'Низкое качество прогноза. Требуются значительные улучшения.',
                'action': 'Критически пересмотрите подход к прогнозированию.',
                'tips': ['Проверьте качество входных данных', 'Рассмотрите использование ансамбля моделей']
            },
            'critical': {
                'general': 'Критически низкое качество прогноза.',
                'action': 'Срочно требуется полный пересмотр методологии.',
                'tips': ['Проведите аудит данных', 'Привлеките экспертов для анализа']
            }
        }
        
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
            'model_recommendations': [],
            'rank_based_recommendations': {}
        }
        
        # Анализ метрик и генерация рекомендаций по рангам
        for metric_type in ['revenue', 'traffic', 'check']:
            mape_key = f'MAPE_{metric_type}'
            rmse_key = f'RMSE_{metric_type}'
            
            if mape_key in metrics and metrics[mape_key] is not None:
                mape_value = metrics[mape_key]
                quality = self._get_mape_quality(mape_value)
                
                # Добавляем рекомендации на основе рангов
                rank_rec = self.rank_recommendations.get(quality, {})
                recommendations['rank_based_recommendations'][metric_type] = {
                    'rank': quality,
                    'rank_rus': self._get_quality_rus(quality),
                    'general': rank_rec.get('general', ''),
                    'action': rank_rec.get('action', ''),
                    'tips': rank_rec.get('tips', [])
                }
                
                # Генерируем детальные рекомендации для проблемных метрик
                if quality in ['acceptable', 'warning', 'poor', 'critical', 'catastrophic']:
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
        """Определение качества прогноза по MAPE с расширенными рангами"""
        if mape <= self.mape_thresholds['excellent']:
            return 'excellent'
        elif mape <= self.mape_thresholds['good']:
            return 'good'
        elif mape <= self.mape_thresholds['acceptable']:
            return 'acceptable'
        elif mape <= self.mape_thresholds['warning']:
            return 'warning'
        elif mape <= self.mape_thresholds['poor']:
            return 'poor'
        elif mape <= self.mape_thresholds['critical']:
            return 'critical'
        else:
            return 'catastrophic'
    
    def _get_quality_rus(self, quality: str) -> str:
        """Перевод качества на русский"""
        translations = {
            'excellent': 'Отличное',
            'good': 'Хорошее',
            'acceptable': 'Приемлемое',
            'warning': 'Требует внимания',
            'poor': 'Плохое',
            'critical': 'Критическое',
            'catastrophic': 'Катастрофическое'
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
            'acceptable': 'metric-acceptable',
            'warning': 'metric-warning',
            'poor': 'metric-poor',
            'critical': 'metric-critical',
            'catastrophic': 'metric-catastrophic'
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
    
    def get_structured_recommendations_for_metric(self, metric_type: str, mape_value: float, 
                                                 rmse_value: Optional[float] = None) -> dict:
        """
        Получение структурированных рекомендаций для конкретной метрики
        
        Args:
            metric_type: Тип метрики (revenue, traffic, check)
            mape_value: Значение MAPE
            rmse_value: Значение RMSE (опционально)
            
        Returns:
            Словарь со структурированными рекомендациями
        """
        quality = self._get_mape_quality(mape_value)
        rank_rec = self.rank_recommendations.get(quality, {})
        
        metric_name_rus = {
            'revenue': 'выручки',
            'traffic': 'трафика', 
            'check': 'среднего чека'
        }.get(metric_type, metric_type)
        
        # Специфичные рекомендации по типам метрик
        specific_tips = {
            'revenue': {
                'excellent': ['Продолжайте мониторинг трендов', 'Отслеживайте сезонные паттерны'],
                'good': ['Проверьте влияние промо-акций', 'Анализируйте выбросы в данных'],
                'acceptable': ['Уточните прогноз для праздничных дней', 'Проверьте корреляцию с внешними факторами'],
                'warning': ['Проанализируйте аномальные периоды', 'Рассмотрите добавление внешних регрессоров'],
                'poor': ['Проведите глубокий анализ данных', 'Рассмотрите использование ML-моделей'],
                'critical': ['Требуется аудит качества данных', 'Возможны структурные изменения в бизнесе']
            },
            'traffic': {
                'excellent': ['Модель хорошо улавливает паттерны', 'Продолжайте текущий подход'],
                'good': ['Проверьте учет выходных дней', 'Уточните влияние погоды'],
                'acceptable': ['Добавьте учет локальных событий', 'Проверьте данные о конкурентах'],
                'warning': ['Анализируйте дни с аномальным трафиком', 'Учтите изменения в расписании'],
                'poor': ['Проверьте качество подсчета трафика', 'Возможны проблемы с учетом'],
                'critical': ['Требуется валидация системы учета', 'Проверьте корректность данных']
            },
            'check': {
                'excellent': ['Стабильная ценовая политика работает', 'Клиенты предсказуемы'],
                'good': ['Отслеживайте изменения в меню', 'Мониторьте средний состав заказа'],
                'acceptable': ['Проверьте влияние новых позиций', 'Анализируйте структуру продаж'],
                'warning': ['Возможны изменения в поведении клиентов', 'Проверьте ценовую политику'],
                'poor': ['Анализируйте изменения в ассортименте', 'Проверьте корректность расчета'],
                'critical': ['Возможны ошибки в данных', 'Требуется ревизия методологии']
            }
        }
        
        tips = rank_rec.get('tips', [])
        specific = specific_tips.get(metric_type, {}).get(quality, [])
        
        return {
            'metric_type': metric_type,
            'metric_name_rus': metric_name_rus,
            'mape_value': mape_value,
            'quality': quality,
            'quality_rus': self._get_quality_rus(quality),
            'color_class': self.get_metric_color_class(mape_value),
            'general_recommendation': rank_rec.get('general', ''),
            'action_required': rank_rec.get('action', ''),
            'general_tips': tips,
            'specific_tips': specific,
            'all_tips': tips + specific,
            'priority': self._get_recommendation_priority(quality)
        }
    
    def _get_recommendation_priority(self, quality: str) -> int:
        """Получение приоритета рекомендации"""
        priority_map = {
            'excellent': 0,
            'good': 1,
            'acceptable': 2,
            'warning': 3,
            'poor': 4,
            'critical': 5,
            'catastrophic': 6
        }
        return priority_map.get(quality, 3)