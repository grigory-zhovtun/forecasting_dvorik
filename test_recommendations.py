#!/usr/bin/env python
"""
Тестовый скрипт для проверки системы рекомендаций
"""

import sys
from src.utils.recommendations import RecommendationEngine

def test_recommendations():
    """Тестирование системы рекомендаций"""
    
    # Создаем движок рекомендаций
    config = {
        'metrics_thresholds': {
            'mape': {
                'excellent': 5,
                'good': 10,
                'acceptable': 15,
                'warning': 20,
                'poor': 30,
                'critical': 40
            }
        }
    }
    
    rec_engine = RecommendationEngine(config)
    
    # Тестовые значения MAPE
    test_values = [3, 8, 12, 18, 25, 35, 45]
    
    print("Тестирование определения рангов качества:")
    print("-" * 50)
    
    for mape in test_values:
        quality = rec_engine._get_mape_quality(mape)
        quality_rus = rec_engine._get_quality_rus(quality)
        color_class = rec_engine.get_metric_color_class(mape)
        
        print(f"MAPE: {mape}% -> Ранг: {quality} ({quality_rus}) -> CSS: {color_class}")
        
        # Проверяем, что ранг существует в rank_recommendations
        if quality not in rec_engine.rank_recommendations and quality not in ['catastrophic', 'unknown']:
            print(f"  ⚠️  ВНИМАНИЕ: Ранг '{quality}' отсутствует в rank_recommendations!")
    
    print("\n" + "=" * 50)
    print("\nТестирование структурированных рекомендаций:")
    print("-" * 50)
    
    # Тестируем для разных метрик и значений
    test_cases = [
        ('revenue', 3),    # excellent
        ('traffic', 12),   # acceptable
        ('check', 25),     # poor
        ('revenue', 45)    # catastrophic
    ]
    
    for metric_type, mape_value in test_cases:
        print(f"\nМетрика: {metric_type}, MAPE: {mape_value}%")
        try:
            rec = rec_engine.get_structured_recommendations_for_metric(metric_type, mape_value)
            print(f"  Качество: {rec['quality_rus']}")
            print(f"  Общая рекомендация: {rec['general_recommendation']}")
            print(f"  Требуемое действие: {rec['action_required']}")
            print(f"  Советы: {', '.join(rec['all_tips'][:2])}")
        except Exception as e:
            print(f"  ❌ ОШИБКА: {e}")
    
    print("\n" + "=" * 50)
    print("\nТестирование полных рекомендаций для кафе:")
    print("-" * 50)
    
    # Тестовые метрики для кафе
    test_metrics = {
        'MAPE_revenue': 12.5,
        'MAPE_traffic': 8.3,
        'MAPE_check': 15.7,
        'MAE_revenue': 5000,
        'MAE_traffic': 50,
        'MAE_check': 100,
        'RMSE_revenue': 7500,
        'RMSE_traffic': 75,
        'RMSE_check': 150,
        'R2_revenue': 0.85,
        'R2_traffic': 0.92,
        'R2_check': 0.78
    }
    
    try:
        recs = rec_engine.get_recommendations_for_cafe('Тестовое кафе', test_metrics, 'prophet')
        print(f"Получены рекомендации для кафе:")
        print(f"  Общих рекомендаций: {len(recs['general_recommendations'])}")
        print(f"  Рекомендаций по параметрам: {len(recs['parameter_recommendations'])}")
        print(f"  Рекомендаций по моделям: {len(recs['model_recommendations'])}")
        print(f"  Рангов по метрикам: {len(recs['rank_based_recommendations'])}")
        
        for metric, rank_data in recs['rank_based_recommendations'].items():
            print(f"    {metric}: {rank_data['rank']} ({rank_data['rank_rus']})")
            
    except Exception as e:
        print(f"❌ ОШИБКА при получении рекомендаций: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_recommendations()