"""
Веб-приложение для прогнозирования выручки сети кафе "Пироговый Дворик"
Версия 3.0 - модульная архитектура с расширенным функционалом
"""

from flask import Flask
import yaml
import logging
import warnings
from pathlib import Path
import os

# Импорт модулей приложения
from src.models.data_loader import DataLoader
from src.models.forecast_engine import ForecastEngine
from src.controllers.views import ForecastViews

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Загрузка конфигурации
def load_config(config_path='config.yaml'):
    """Загрузка конфигурации из файла"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Ошибка загрузки конфигурации: {e}")
        # Возвращаем минимальную конфигурацию по умолчанию
        return {
            'server': {'host': '0.0.0.0', 'port': 5000, 'debug': True},
            'data': {'facts_file': 'facts.xlsx'},
            'database': {'path': 'forecasting_db.sqlite'}
        }

# Инициализация компонентов
config = load_config()
data_loader = DataLoader(config)
forecast_engine = ForecastEngine(config, data_loader)
views = ForecastViews(app, config, data_loader, forecast_engine)

if __name__ == '__main__':
    # Создание необходимых директорий
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('utils', exist_ok=True)
    
    # Запуск сервера
    server_config = config.get('server', {})
    app.run(
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 5000),
        debug=server_config.get('debug', True),
        threaded=True
    )