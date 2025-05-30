# Инструкция по развертыванию на удаленном ПК

## Требования
- Python 3.8+ 
- pip

## Установка

1. **Клонируйте репозиторий или скопируйте файлы проекта**

2. **Создайте виртуальное окружение:**
```bash
python -m venv venv
```

3. **Активируйте виртуальное окружение:**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

4. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

5. **Убедитесь, что следующие файлы находятся в корне проекта:**
- `facts.xlsx` - данные о продажах
- `holidays_df.xlsx` - праздники
- `prophet_holidays_2020_2027.csv` - праздники для Prophet
- `russian_calendar_2020_2027.csv` - российский календарь

6. **Создайте директорию data и поместите в нее:**
- `data/passport.xlsx` - паспорт кафе

7. **Проверьте config.yaml (если есть) или будут использованы настройки по умолчанию**

## Запуск

```bash
python app.py
```

Приложение будет доступно по адресу: http://localhost:5000

## Возможные проблемы

### Ошибка импорта модулей
Убедитесь, что вы находитесь в корневой директории проекта при запуске app.py

### Ошибка с prophet/cmdstan
Если возникают проблемы с установкой prophet:
```bash
pip install pystan==2.19.1.1
pip install prophet --no-deps
pip install -r requirements.txt --force-reinstall
```

### Ошибка с torch
Для Windows без CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Структура проекта
```
forecasting_dvorik/
├── app.py
├── config.yaml
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── data_loader.py
│   │   ├── forecast_engine.py
│   │   └── holidays.py
│   ├── controllers/
│   │   └── views.py
│   └── utils/
│       └── recommendations.py
├── data/
│   └── passport.xlsx
├── templates/
│   └── index.html
├── static/
└── файлы данных (*.xlsx, *.csv)
```