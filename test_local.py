#!/usr/bin/env python3
"""
Скрипт для тестирования основных функций локально
"""

import pandas as pd
from src.models.data_loader import DataLoader

# Тест загрузки данных паспорта
print("Тестирование загрузки данных паспорта...")
try:
    config = {
        'data': {
            'facts_file': 'facts.xlsx',
            'holidays_file': 'holidays_df.xlsx',
            'prophet_holidays_file': 'prophet_holidays_2020_2027.csv',
            'russian_calendar_file': 'russian_calendar_2020_2027.csv'
        },
        'excluded_cafes': ['ПД-15', 'ПД-20', 'ПД-25', 'Кейтеринг']
    }
    
    data_loader = DataLoader(config)
    passport_df = data_loader.load_passport_data()
    
    if passport_df.empty:
        print("❌ Данные паспорта пустые!")
    else:
        print(f"✅ Загружено {len(passport_df)} записей паспорта")
        print("Колонки:", passport_df.columns.tolist())
        print("\nПервые записи:")
        print(passport_df.head())
        
except Exception as e:
    print(f"❌ Ошибка: {e}")

# Тест загрузки фактов
print("\n\nТестирование загрузки фактов...")
try:
    facts_df = data_loader.load_data()
    cafes = data_loader.get_cafes_list()
    
    print(f"✅ Загружено {len(facts_df)} записей")
    print(f"✅ Найдено кафе: {cafes}")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")