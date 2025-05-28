"""
Модуль управления пресетами и корректировками
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class PresetManager:
    """Класс для управления пресетами и корректировками"""
    
    def __init__(self, config: dict):
        """
        Инициализация менеджера пресетов
        
        Args:
            config: Конфигурация приложения
        """
        self.config = config
        self.db_path = config.get('database', {}).get('path', 'forecasting_db.sqlite')
        self._init_database()
    
    def _init_database(self):
        """Инициализация базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Таблица пресетов
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS presets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        cafes TEXT NOT NULL,
                        params TEXT NOT NULL,
                        adjustments TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Таблица корректировок
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS adjustments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cafe TEXT NOT NULL,
                        date_from DATE NOT NULL,
                        date_to DATE NOT NULL,
                        metric TEXT NOT NULL,
                        coefficient REAL NOT NULL,
                        reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Индексы
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_presets_name ON presets(name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_adjustments_cafe ON adjustments(cafe)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_adjustments_dates ON adjustments(date_from, date_to)")
                
                conn.commit()
                logger.info("База данных инициализирована")
                
        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            raise
    
    def save_preset(self, name: str, cafes: List[str], params: dict, adjustments: List[dict] = None) -> int:
        """
        Сохранение пресета
        
        Args:
            name: Название пресета
            cafes: Список выбранных кафе
            params: Параметры прогнозирования
            adjustments: Список корректировок
            
        Returns:
            ID созданного пресета
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Проверяем, существует ли пресет с таким именем
                cursor.execute("SELECT id FROM presets WHERE name = ?", (name,))
                existing = cursor.fetchone()
                
                cafes_json = json.dumps(cafes)
                params_json = json.dumps(params)
                adjustments_json = json.dumps(adjustments or [])
                
                if existing:
                    # Обновляем существующий
                    cursor.execute("""
                        UPDATE presets 
                        SET cafes = ?, params = ?, adjustments = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (cafes_json, params_json, adjustments_json, existing[0]))
                    preset_id = existing[0]
                    logger.info(f"Пресет '{name}' обновлен")
                else:
                    # Создаем новый
                    cursor.execute("""
                        INSERT INTO presets (name, cafes, params, adjustments)
                        VALUES (?, ?, ?, ?)
                    """, (name, cafes_json, params_json, adjustments_json))
                    preset_id = cursor.lastrowid
                    logger.info(f"Пресет '{name}' создан")
                
                conn.commit()
                return preset_id
                
        except Exception as e:
            logger.error(f"Ошибка сохранения пресета: {e}")
            raise
    
    def load_preset(self, preset_id: int) -> Optional[dict]:
        """
        Загрузка пресета по ID
        
        Args:
            preset_id: ID пресета
            
        Returns:
            Словарь с данными пресета или None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM presets WHERE id = ?
                """, (preset_id,))
                
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row['id'],
                        'name': row['name'],
                        'cafes': json.loads(row['cafes']),
                        'params': json.loads(row['params']),
                        'adjustments': json.loads(row['adjustments'] or '[]'),
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Ошибка загрузки пресета: {e}")
            return None
    
    def load_preset_by_name(self, name: str) -> Optional[dict]:
        """
        Загрузка пресета по имени
        
        Args:
            name: Название пресета
            
        Returns:
            Словарь с данными пресета или None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM presets WHERE name = ?
                """, (name,))
                
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row['id'],
                        'name': row['name'],
                        'cafes': json.loads(row['cafes']),
                        'params': json.loads(row['params']),
                        'adjustments': json.loads(row['adjustments'] or '[]'),
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Ошибка загрузки пресета по имени: {e}")
            return None
    
    def list_presets(self) -> List[dict]:
        """
        Получение списка всех пресетов
        
        Returns:
            Список пресетов
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, name, created_at, updated_at 
                    FROM presets 
                    ORDER BY updated_at DESC
                """)
                
                rows = cursor.fetchall()
                
                return [
                    {
                        'id': row['id'],
                        'name': row['name'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Ошибка получения списка пресетов: {e}")
            return []
    
    def delete_preset(self, preset_id: int) -> bool:
        """
        Удаление пресета
        
        Args:
            preset_id: ID пресета
            
        Returns:
            True если успешно удален
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Ошибка удаления пресета: {e}")
            return False
    
    def save_adjustment(self, cafe: str, date_from: str, date_to: str, 
                       metric: str, coefficient: float, reason: str = None) -> int:
        """
        Сохранение корректировки
        
        Args:
            cafe: Название кафе или 'ALL'
            date_from: Дата начала
            date_to: Дата окончания
            metric: Тип метрики (traffic, check, both)
            coefficient: Коэффициент корректировки
            reason: Причина корректировки
            
        Returns:
            ID созданной корректировки
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO adjustments (cafe, date_from, date_to, metric, coefficient, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (cafe, date_from, date_to, metric, coefficient, reason))
                
                conn.commit()
                adjustment_id = cursor.lastrowid
                
                logger.info(f"Корректировка создана: {cafe} {date_from} - {date_to}")
                return adjustment_id
                
        except Exception as e:
            logger.error(f"Ошибка сохранения корректировки: {e}")
            raise
    
    def list_adjustments(self, cafe: str = None, start_date: str = None, 
                        end_date: str = None) -> List[dict]:
        """
        Получение списка корректировок
        
        Args:
            cafe: Фильтр по кафе
            start_date: Фильтр по начальной дате
            end_date: Фильтр по конечной дате
            
        Returns:
            Список корректировок
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM adjustments WHERE 1=1"
                params = []
                
                if cafe:
                    query += " AND (cafe = ? OR cafe = 'ALL')"
                    params.append(cafe)
                
                if start_date:
                    query += " AND date_to >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND date_from <= ?"
                    params.append(end_date)
                
                query += " ORDER BY date_from DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [
                    {
                        'id': row['id'],
                        'cafe': row['cafe'],
                        'date_from': row['date_from'],
                        'date_to': row['date_to'],
                        'metric': row['metric'],
                        'coefficient': row['coefficient'],
                        'reason': row['reason'],
                        'created_at': row['created_at']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Ошибка получения списка корректировок: {e}")
            return []
    
    def delete_adjustment(self, adjustment_id: int) -> bool:
        """
        Удаление корректировки
        
        Args:
            adjustment_id: ID корректировки
            
        Returns:
            True если успешно удалена
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM adjustments WHERE id = ?", (adjustment_id,))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Ошибка удаления корректировки: {e}")
            return False
    
    def update_adjustment(self, adjustment_id: int, **kwargs) -> bool:
        """
        Обновление корректировки
        
        Args:
            adjustment_id: ID корректировки
            **kwargs: Поля для обновления
            
        Returns:
            True если успешно обновлена
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Строим запрос динамически
                allowed_fields = ['cafe', 'date_from', 'date_to', 'metric', 'coefficient', 'reason']
                update_fields = []
                values = []
                
                for field, value in kwargs.items():
                    if field in allowed_fields:
                        update_fields.append(f"{field} = ?")
                        values.append(value)
                
                if not update_fields:
                    return False
                
                values.append(adjustment_id)
                query = f"UPDATE adjustments SET {', '.join(update_fields)} WHERE id = ?"
                
                cursor.execute(query, values)
                conn.commit()
                
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Ошибка обновления корректировки: {e}")
            return False
    
    def get_active_adjustments(self, date: str) -> List[dict]:
        """
        Получение активных корректировок на указанную дату
        
        Args:
            date: Дата для проверки
            
        Returns:
            Список активных корректировок
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM adjustments
                    WHERE date_from <= ? AND date_to >= ?
                    ORDER BY created_at DESC
                """, (date, date))
                
                rows = cursor.fetchall()
                
                return [
                    {
                        'id': row['id'],
                        'cafe': row['cafe'],
                        'date_from': row['date_from'],
                        'date_to': row['date_to'],
                        'metric': row['metric'],
                        'coefficient': row['coefficient'],
                        'reason': row['reason']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Ошибка получения активных корректировок: {e}")
            return []
    
    def export_presets(self, file_path: str) -> bool:
        """
        Экспорт всех пресетов в JSON файл
        
        Args:
            file_path: Путь к файлу для экспорта
            
        Returns:
            True если успешно экспортировано
        """
        try:
            presets = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM presets")
                rows = cursor.fetchall()
                
                for row in rows:
                    presets.append({
                        'name': row['name'],
                        'cafes': json.loads(row['cafes']),
                        'params': json.loads(row['params']),
                        'adjustments': json.loads(row['adjustments'] or '[]')
                    })
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(presets, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Экспортировано {len(presets)} пресетов в {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта пресетов: {e}")
            return False
    
    def import_presets(self, file_path: str) -> int:
        """
        Импорт пресетов из JSON файла
        
        Args:
            file_path: Путь к файлу для импорта
            
        Returns:
            Количество импортированных пресетов
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                presets = json.load(f)
            
            imported = 0
            for preset in presets:
                try:
                    self.save_preset(
                        name=preset['name'],
                        cafes=preset['cafes'],
                        params=preset['params'],
                        adjustments=preset.get('adjustments', [])
                    )
                    imported += 1
                except Exception as e:
                    logger.error(f"Ошибка импорта пресета '{preset.get('name', 'Unknown')}': {e}")
            
            logger.info(f"Импортировано {imported} из {len(presets)} пресетов")
            return imported
            
        except Exception as e:
            logger.error(f"Ошибка импорта пресетов: {e}")
            return 0