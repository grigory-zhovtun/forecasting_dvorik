"""
Модуль загрузки и подготовки данных для прогнозирования
"""

import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и подготовки данных"""
    
    def __init__(self, config_or_path = "config.yaml"):
        """
        Инициализация загрузчика данных
        
        Args:
            config_or_path: Путь к файлу конфигурации или словарь с конфигурацией
        """
        if isinstance(config_or_path, dict):
            self.config = config_or_path
        else:
            self.config = self._load_config(config_or_path)
        self.data_paths = self.config.get('data', {
            'facts_file': 'facts.xlsx',
            'holidays_file': 'holidays_df.xlsx',
            'prophet_holidays_file': 'prophet_holidays_2020_2027.csv',
            'russian_calendar_file': 'russian_calendar_2020_2027.csv'
        })
        self.excluded_cafes = self.config.get('excluded_cafes', ['ПД-15', 'ПД-20', 'ПД-25', 'Кейтеринг', 'Кафе на Садовой', 'Кафе на Невском'])
        self._cache = {}
    
    def clear_cache(self):
        """Очистка кэша для перезагрузки данных"""
        logger.info("Очистка кэша данных")
        self._cache = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            # Возвращаем конфигурацию по умолчанию
            return {
                'data': {
                    'facts_file': 'facts.xlsx',
                    'holidays_file': 'holidays_df.xlsx',
                    'prophet_holidays_file': 'prophet_holidays_2020_2027.csv',
                    'russian_calendar_file': 'russian_calendar_2020_2027.csv'
                },
                'excluded_cafes': ['ПД-15', 'ПД-20', 'ПД-25', 'Кейтеринг']
            }
    
    def load_facts_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Загрузка фактических данных о продажах
        
        Args:
            force_reload: Если True, данные загружаются заново, игнорируя кэш
            
        Returns:
            DataFrame с фактическими данными
        """
        if not force_reload and 'facts' in self._cache:
            return self._cache['facts']
            
        try:
            # Пробуем загрузить по абсолютному пути из конфига
            df = pd.read_excel(self.data_paths['facts_file'])
            logger.info(f"Данные загружены из {self.data_paths['facts_file']}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить данные по пути из конфига: {e}")
            # Пробуем локальный файл
            local_path = 'facts.xlsx'
            try:
                df = pd.read_excel(local_path)
                logger.info(f"Данные загружены из локального файла {local_path}")
            except Exception as e2:
                logger.error(f"Не удалось загрузить данные: {e2}")
                raise
        
        # Фильтрация и обработка данных
        df = self._process_facts_data(df)
        self._cache['facts'] = df
        
        return df
    
    def _process_facts_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка фактических данных
        
        Args:
            df: Исходный DataFrame
            
        Returns:
            Обработанный DataFrame
        """
        # Исключаем ненужные кафе
        df = df[~df['Кафе'].isin(self.excluded_cafes)]
        
        # Удаляем строки с пропущенными значениями
        df = df.dropna(subset=['Дата', 'Тр', 'Чек'])
        
        # Преобразуем дату
        df['Дата'] = pd.to_datetime(df['Дата'])
        
        # Рассчитываем выручку
        df['Выручка'] = df['Тр'] * df['Чек']
        
        logger.info(f"Обработано {len(df)} записей для {df['Кафе'].nunique()} кафе")
        
        return df
    
    def load_holidays(self) -> pd.DataFrame:
        """
        Загрузка данных о праздниках
        
        Returns:
            DataFrame с праздниками для Prophet
        """
        if 'holidays' in self._cache:
            return self._cache['holidays']
            
        holidays_list = []
        
        # 1. Загружаем пользовательские праздники
        custom_holidays = self._load_custom_holidays()
        if not custom_holidays.empty:
            holidays_list.append(custom_holidays)
        
        # 2. Загружаем российские праздники
        ru_holidays = self._load_russian_holidays()
        if not ru_holidays.empty:
            holidays_list.append(ru_holidays)
        
        # Объединяем все праздники
        if holidays_list:
            holidays = pd.concat(holidays_list, ignore_index=True).drop_duplicates()
        else:
            holidays = pd.DataFrame(columns=['ds', 'holiday'])
            
        logger.info(f"Загружено {len(holidays)} праздников")
        self._cache['holidays'] = holidays
        
        return holidays
    
    def _load_custom_holidays(self) -> pd.DataFrame:
        """Загрузка пользовательских праздников"""
        try:
            # Пробуем путь из конфига
            df = pd.read_excel(self.data_paths['holidays_file'])
            logger.info(f"Пользовательские праздники загружены из {self.data_paths['holidays_file']}")
        except Exception:
            # Пробуем локальный файл
            try:
                df = pd.read_excel('holidays_df.xlsx')
                logger.info("Пользовательские праздники загружены из локального файла")
            except Exception:
                logger.warning("Не удалось загрузить пользовательские праздники")
                return pd.DataFrame(columns=['ds', 'holiday'])
        
        # Убеждаемся что есть нужные колонки
        if 'ds' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'ds'})
            
        return df[['ds', 'holiday']]
    
    def _load_russian_holidays(self) -> pd.DataFrame:
        """Загрузка российских праздников"""
        prophet_file = self.data_paths.get('prophet_holidays_file', 'prophet_holidays_2020_2027.csv')
        
        # Пробуем несколько вариантов путей
        paths_to_try = [
            prophet_file,
            'prophet_holidays_2020_2027.csv',
            'prophet_holidays_2020_2026.csv'
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    df['ds'] = pd.to_datetime(df['ds'])
                    logger.info(f"Российские праздники загружены из {path}")
                    return df[['ds', 'holiday']]
                except Exception as e:
                    logger.warning(f"Ошибка при загрузке праздников из {path}: {e}")
        
        # Если файлы не найдены, создаем базовый набор
        logger.warning("Не удалось загрузить файлы с праздниками, создаем базовый набор")
        return self._create_basic_holidays()
    
    def _create_basic_holidays(self) -> pd.DataFrame:
        """Создание базового набора российских праздников"""
        holidays = []
        current_year = datetime.now().year
        
        for year in range(2020, current_year + 3):
            # Основные российские праздники
            holidays.extend([
                {'ds': f'{year}-01-01', 'holiday': 'Новый год'},
                {'ds': f'{year}-01-07', 'holiday': 'Рождество Христово'},
                {'ds': f'{year}-02-23', 'holiday': 'День защитника Отечества'},
                {'ds': f'{year}-03-08', 'holiday': 'Международный женский день'},
                {'ds': f'{year}-05-01', 'holiday': 'Праздник Весны и Труда'},
                {'ds': f'{year}-05-09', 'holiday': 'День Победы'},
                {'ds': f'{year}-06-12', 'holiday': 'День России'},
                {'ds': f'{year}-11-04', 'holiday': 'День народного единства'},
            ])
        
        df = pd.DataFrame(holidays)
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    
    def get_cafes_list(self) -> List[str]:
        """
        Получение списка всех кафе
        
        Returns:
            Список названий кафе
        """
        df = self.load_facts_data()
        return sorted(df['Кафе'].unique().tolist())
    
    def get_date_range(self) -> Dict[str, str]:
        """
        Получение диапазона дат в данных
        
        Returns:
            Словарь с минимальной и максимальной датами
        """
        df = self.load_facts_data()
        return {
            'min': df['Дата'].min().strftime('%Y-%m-%d'),
            'max': df['Дата'].max().strftime('%Y-%m-%d')
        }
    
    def get_cafe_data(self, cafe: str) -> pd.DataFrame:
        """
        Получение данных для конкретного кафе
        
        Args:
            cafe: Название кафе
            
        Returns:
            DataFrame с данными кафе
        """
        # Всегда перезагружаем данные для получения свежей информации
        df = self.load_facts_data(force_reload=True)
        cafe_df = df[df['Кафе'] == cafe].copy()
        
        # Группировка по датам (если есть дубликаты)
        cafe_df = cafe_df.groupby('Дата').agg({
            'Тр': 'sum',
            'Чек': 'mean',
            'Выручка': 'sum'
        }).reset_index()
        
        return cafe_df
    
    def classify_cafe(self, cafe: str) -> str:
        """
        Классификация кафе по количеству дней работы
        
        Args:
            cafe: Название кафе
            
        Returns:
            Категория кафе: 'mature', 'established', 'developing', 'new'
        """
        cafe_df = self.get_cafe_data(cafe)
        unique_days = cafe_df['Дата'].nunique()
        
        if unique_days >= 365:
            return "mature"
        elif unique_days >= 180:
            return "established"
        elif unique_days >= 90:
            return "developing"
        else:
            return "new"
    
    def remove_outliers(self, df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Удаление выбросов из данных методом IQR
        
        Args:
            df: DataFrame с данными
            column: Название колонки для обработки
            multiplier: Множитель для IQR
            
        Returns:
            DataFrame без выбросов
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        original_count = len(df)
        df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        removed_count = original_count - len(df_filtered)
        
        if removed_count > 0:
            logger.info(f"Удалено {removed_count} выбросов из колонки {column}")
        
        return df_filtered
    
    def load_passport_data(self) -> pd.DataFrame:
        """
        Загрузка данных паспорта кафе
        
        Returns:
            DataFrame с данными паспорта
        """
        try:
            passport_file = os.path.join('data', 'passport.xlsx')
            if os.path.exists(passport_file):
                # Get sheet names before loading all sheets
                try:
                    excel_file = pd.ExcelFile(passport_file)
                    sheet_names = excel_file.sheet_names
                    logger.info(f"Passport Excel - Sheet names: {sheet_names}")
                except Exception as e_sheets:
                    logger.warning(f"Could not read sheet names from {passport_file}: {e_sheets}")
                    # Attempt to load the first sheet by default if reading sheet names fails
                    sheet_names = [0] # Use 0 to load the first sheet

                # Load the first sheet by default, or all if multiple sheets are relevant
                # For now, let's assume the relevant data is in the first sheet.
                # If all sheets need to be processed, this logic should be adjusted.
                passport_df = pd.read_excel(passport_file, sheet_name=sheet_names[0] if isinstance(sheet_names, list) and len(sheet_names)>0 else 0)
                
                logger.info(f"Passport DataFrame shape: {passport_df.shape}")
                logger.info(f"Passport DataFrame columns: {passport_df.columns.tolist()}")
                logger.info(f"Passport DataFrame head:\n{passport_df.head().to_string()}")
                logger.info(f"Загружены данные паспорта для {len(passport_df)} кафе")
                return passport_df
            else:
                logger.warning(f"Файл паспорта '{passport_file}' не найден")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных паспорта из файла '{passport_file}': {e}")
            return pd.DataFrame()
    
    def get_cafe_passport(self, cafe_name: str) -> dict:
        """
        Получение данных паспорта для конкретного кафе
        
        Args:
            cafe_name: Название кафе
            
        Returns:
            Словарь с данными паспорта
        """
        passport_df = self.load_passport_data()
        if passport_df.empty:
            return {}
        
        cafe_data = passport_df[passport_df['cafe'] == cafe_name]
        if cafe_data.empty:
            return {}
        
        return cafe_data.iloc[0].to_dict()
    
    def prepare_for_forecast(self, cafe_df: pd.DataFrame, remove_outliers: bool = True, 
                           outlier_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Подготовка данных кафе для прогнозирования
        
        Args:
            cafe_df: DataFrame с данными кафе
            remove_outliers: Флаг удаления выбросов
            outlier_multiplier: Множитель для определения выбросов
            
        Returns:
            Подготовленный DataFrame
        """
        df = cafe_df.copy()
        
        # Удаление выбросов если требуется
        if remove_outliers:
            df = self.remove_outliers(df, 'Тр', outlier_multiplier)
            df = self.remove_outliers(df, 'Чек', outlier_multiplier)
        
        # Заполнение пропущенных дат
        df = df.set_index('Дата').asfreq('D').reset_index()
        
        # Заполнение пропущенных значений
        df['Тр'] = df['Тр'].fillna(df['Тр'].mean())
        df['Чек'] = df['Чек'].fillna(df['Чек'].mean())
        df['Выручка'] = df['Тр'] * df['Чек']
        
        return df
    
    def clear_cache(self):
        """Очистка кэша данных"""
        self._cache.clear()
        logger.info("Кэш данных очищен")