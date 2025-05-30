# Исправления для удаленного ПК

## 1. Исправление паспорта кафе

Проверьте, что файл `data/passport.xlsx` существует и содержит правильные данные.

Если файла нет, создайте его с помощью скрипта:

```python
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Создаем директорию data если её нет
os.makedirs('data', exist_ok=True)

# Список кафе из facts.xlsx (должны совпадать с названиями в facts.xlsx!)
cafes = [
    "Кафе на Арбате", "Кафе на Тверской", "Кафе на Пушкинской",
    "Кафе в ТЦ Европейский", "Кафе на Маяковской", "Кафе на Цветном"
]

# Генерация данных паспорта
data = []
for cafe in cafes:
    days_ago = random.randint(365, 365*5)
    open_date = datetime.now() - timedelta(days=days_ago)
    
    location_types = ["Бизнес центр", "Спальный район", "Центр города", "ТРЦ", "Улица с высоким трафиком"]
    location = random.choice(location_types)
    
    seats = random.randint(30, 120)
    staff = random.randint(8, 25)
    area = random.randint(80, 300)
    
    data.append({
        "cafe": cafe,
        "open_date": open_date.strftime("%Y-%m-%d"),
        "seats": seats,
        "location_type": location,
        "staff_count": staff,
        "area_sqm": area,
        "parking": random.choice(["Да", "Нет"]),
        "delivery": random.choice(["Да", "Нет"]),
        "kitchen_type": "Пироги и выпечка",
        "work_hours": "08:00-22:00"
    })

# Создание DataFrame и сохранение в Excel
df = pd.DataFrame(data)
df.to_excel("data/passport.xlsx", index=False)
print("Файл passport.xlsx создан в директории data/")
print("Созданы данные для кафе:")
for cafe in cafes:
    print(f"  - {cafe}")
```

## 2. Проверка структуры проекта

Убедитесь, что структура проекта выглядит так:
```
forecasting_dvorik/
├── app.py
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── forecast_engine.py
│   │   └── holidays.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   └── views.py
│   └── utils/
│       ├── __init__.py
│       └── recommendations.py
├── data/
│   └── passport.xlsx
├── templates/
│   └── index.html
├── static/
└── Файлы данных в корне:
    ├── facts.xlsx
    ├── holidays_df.xlsx
    ├── prophet_holidays_2020_2027.csv
    └── russian_calendar_2020_2027.csv
```

## 3. Тестирование локально

Создайте файл test_debug.py в корне проекта:

```python
import os
import pandas as pd

print("Проверка структуры проекта...")

# Проверка наличия файлов
files_to_check = [
    "facts.xlsx",
    "holidays_df.xlsx",
    "prophet_holidays_2020_2027.csv",
    "russian_calendar_2020_2027.csv",
    "data/passport.xlsx",
    "src/__init__.py",
    "src/models/__init__.py",
    "src/models/data_loader.py",
    "src/controllers/__init__.py",
    "src/controllers/views.py",
    "src/utils/__init__.py",
    "src/utils/recommendations.py"
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - НЕ НАЙДЕН!")

# Проверка загрузки паспорта
print("\nПроверка данных паспорта...")
try:
    passport_df = pd.read_excel("data/passport.xlsx")
    print(f"✅ Загружено {len(passport_df)} записей")
    print("Кафе в паспорте:")
    for cafe in passport_df['cafe'].values:
        print(f"  - {cafe}")
except Exception as e:
    print(f"❌ Ошибка загрузки паспорта: {e}")

# Проверка фактов
print("\nПроверка данных фактов...")
try:
    facts_df = pd.read_excel("facts.xlsx")
    cafes_in_facts = facts_df['Кафе'].unique()
    print(f"✅ Найдено {len(cafes_in_facts)} кафе в facts.xlsx:")
    for cafe in cafes_in_facts:
        print(f"  - {cafe}")
except Exception as e:
    print(f"❌ Ошибка загрузки фактов: {e}")
```

## 4. Запуск приложения

После всех проверок запустите:
```bash
python app.py
```

## 5. Если все еще есть проблемы

1. Откройте браузер в режиме инкогнито
2. Очистите кэш браузера (Ctrl+F5)
3. Откройте консоль разработчика (F12) и проверьте ошибки JavaScript

## 6. Проверка корректировок

После расчета прогноза:
1. Перейдите на вкладку "Корректировки"
2. Заполните все поля:
   - Кафе (или "Все кафе")
   - Показатель (выручка/трафик/средний чек)
   - Тип (процент или абсолютное значение)
   - Значение (например, 10 для 10%)
   - Даты начала и конца периода
3. Нажмите "Применить"

Если появляется ошибка, проверьте консоль браузера (F12) для деталей.