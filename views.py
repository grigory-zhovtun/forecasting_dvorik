"""
Модуль с маршрутами и представлениями Flask
"""

from flask import jsonify, request, send_file, render_template
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import threading
import queue
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ForecastViews:
    """Класс для управления представлениями и маршрутами"""
    
    def __init__(self, app, config: dict, data_loader, forecast_engine, preset_manager):
        """
        Инициализация представлений
        
        Args:
            app: Flask приложение
            config: Конфигурация
            data_loader: Экземпляр DataLoader
            forecast_engine: Экземпляр ForecastEngine
            preset_manager: Экземпляр PresetManager
        """
        self.app = app
        self.config = config
        self.data_loader = data_loader
        self.forecast_engine = forecast_engine
        self.preset_manager = preset_manager
        
        # Кэши и очереди
        self.forecast_cache = {}
        self.metrics_cache = {}
        self.progress_queue = queue.Queue()
        self.current_task = None
        self.cancel_flag = threading.Event()
        
        # Регистрация маршрутов
        self._register_routes()
    
    def _register_routes(self):
        """Регистрация всех маршрутов"""
        # Главная страница
        self.app.route('/')(self.index)
        
        # API маршруты
        self.app.route('/api/get_initial_data')(self.get_initial_data)
        self.app.route('/api/forecast', methods=['POST'])(self.forecast)
        self.app.route('/api/progress')(self.get_progress)
        self.app.route('/api/cancel_forecast', methods=['POST'])(self.cancel_forecast)
        self.app.route('/api/get_forecast_data')(self.get_forecast_data)
        self.app.route('/api/export_excel')(self.export_excel)
        self.app.route('/api/get_recommendations', methods=['POST'])(self.get_recommendations)
        
        # Маршруты для пресетов
        self.app.route('/api/presets/list')(self.list_presets)
        self.app.route('/api/presets/load/<int:preset_id>')(self.load_preset)
        self.app.route('/api/presets/save', methods=['POST'])(self.save_preset)
        self.app.route('/api/presets/delete/<int:preset_id>', methods=['DELETE'])(self.delete_preset)
        
        # Маршруты для корректировок
        self.app.route('/api/adjustments/list')(self.list_adjustments)
        self.app.route('/api/adjustments/save', methods=['POST'])(self.save_adjustment)
        self.app.route('/api/adjustments/delete/<int:adjustment_id>', methods=['DELETE'])(self.delete_adjustment)
    
    def index(self):
        """Главная страница"""
        return render_template('index.html')
    
    def get_initial_data(self):
        """Получение начальных данных"""
        try:
            cafes = self.data_loader.get_cafes_list()
            date_range = self.data_loader.get_date_range()
            
            # Загружаем дефолтные параметры из конфига
            default_params = self.config.get('forecast_defaults', {})
            
            return jsonify({
                'cafes': cafes,
                'date_range': date_range,
                'default_params': {
                    'forecast_horizon': default_params.get('horizon', 365),
                    'changepoint_prior_scale': default_params.get('prophet', {}).get('changepoint_prior_scale', 0.05),
                    'seasonality_prior_scale': default_params.get('prophet', {}).get('seasonality_prior_scale', 10.0),
                    'outlier_multiplier': default_params.get('outliers', {}).get('multiplier', 1.5),
                    'remove_outliers': default_params.get('outliers', {}).get('remove', True),
                    'yearly_seasonality': default_params.get('prophet', {}).get('yearly_seasonality', True),
                    'weekly_seasonality': default_params.get('prophet', {}).get('weekly_seasonality', True),
                    'daily_seasonality': default_params.get('prophet', {}).get('daily_seasonality', False),
                    'model_type': 'prophet',
                    'confidence_interval': default_params.get('confidence_interval', 0.95),
                    'train_split': default_params.get('train_split', 0.8),
                    'auto_tune': default_params.get('auto_tune', False)
                }
            })
        except Exception as e:
            logger.error(f"Ошибка при получении начальных данных: {e}")
            return jsonify({'error': str(e)}), 500
    
    def forecast(self):
        """Запуск прогнозирования"""
        params = request.json
        selected_cafes = params.get('cafes', [])
        
        if not selected_cafes:
            return jsonify({'error': 'Не выбрано ни одного кафе'}), 400
        
        # Сбрасываем флаг отмены
        self.cancel_flag.clear()
        
        # Очищаем очередь прогресса
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except:
                break
        
        # Запуск прогнозирования в отдельном потоке
        def run_forecast():
            def progress_callback(current, total, message):
                self.progress_queue.put({
                    'current': current,
                    'total': total,
                    'message': message,
                    'percentage': int((current / total) * 100)
                })
            
            # Очищаем старый кэш метрик
            self.metrics_cache.clear()
            
            result, metrics = self.forecast_engine.forecast_all(
                selected_cafes, 
                params, 
                lambda i, t, m: progress_callback(i, t, m) if not self.cancel_flag.is_set() else None
            )
            
            if not self.cancel_flag.is_set():
                self.forecast_cache['latest'] = result
                self.forecast_cache['params'] = params
                self.metrics_cache.update(metrics)
                self.progress_queue.put({'status': 'complete'})
        
        self.current_task = threading.Thread(target=run_forecast)
        self.current_task.start()
        
        return jsonify({'status': 'started'})
    
    def get_progress(self):
        """Получение прогресса выполнения"""
        try:
            progress = self.progress_queue.get_nowait()
            return jsonify(progress)
        except queue.Empty:
            return jsonify({'status': 'waiting'})
    
    def cancel_forecast(self):
        """Отмена текущего прогнозирования"""
        self.cancel_flag.set()
        
        # Очищаем очередь прогресса
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except:
                break
        
        self.progress_queue.put({'status': 'cancelled'})
        
        return jsonify({'status': 'cancelled'})
    
    def get_forecast_data(self):
        """Получение данных прогноза для визуализации"""
        try:
            if 'latest' not in self.forecast_cache:
                return jsonify({'error': 'Нет данных прогноза'}), 404
            
            df = self.forecast_cache['latest'].copy()
            forecast_params = self.forecast_cache.get('params', {})
            
            if df.empty:
                return jsonify({
                    'data': [],
                    'layout': {'title': 'Нет данных для отображения'},
                    'metrics': {},
                    'summary_data': {}
                })
            
            # Фильтрация по датам
            date_from = request.args.get('date_from') or forecast_params.get('date_filter_from')
            date_to = request.args.get('date_to') or forecast_params.get('date_filter_to')
            
            # Если даты не заданы, используем текущий месяц
            if not date_from and not date_to:
                today = datetime.now()
                date_from = today.replace(day=1).strftime('%Y-%m-%d')
                next_month = today.replace(day=28) + timedelta(days=4)
                date_to = (next_month.replace(day=1) - timedelta(days=1)).strftime('%Y-%m-%d')
            
            if date_from:
                df = df[df['ds'] >= pd.to_datetime(date_from)]
            if date_to:
                df = df[df['ds'] <= pd.to_datetime(date_to)]
            
            # Подготовка данных для графика
            traces = self._prepare_plot_traces(df)
            layout = self._prepare_plot_layout()
            
            # Подготовка сводных данных
            summary_data = self._prepare_summary_data(df)
            
            # Получаем рекомендации для текущих метрик
            recommendations = {}
            if self.metrics_cache:
                from utils.recommendations import RecommendationEngine
                rec_engine = RecommendationEngine(self.config)
                for cafe, metrics in self.metrics_cache.items():
                    recommendations[cafe] = rec_engine.get_recommendations_for_cafe(
                        cafe, metrics, forecast_params.get('model_type', 'prophet')
                    )
            
            return jsonify({
                'data': traces,
                'layout': layout,
                'metrics': self.metrics_cache,
                'params': forecast_params,
                'summary_data': summary_data,
                'recommendations': recommendations
            })
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для графика: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _prepare_plot_traces(self, df: pd.DataFrame) -> List[dict]:
        """Подготовка данных для графика Plotly"""
        traces = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, cafe in enumerate(df['cafe'].unique()):
            cafe_data = df[df['cafe'] == cafe].sort_values('ds')
            color = colors[idx % len(colors)]
            
            # Факт
            fact_data = cafe_data[cafe_data['revenue_fact'].notna()]
            if not fact_data.empty:
                traces.append({
                    'x': fact_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'y': fact_data['revenue_fact'].round(2).tolist(),
                    'name': f'{cafe} - Факт',
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {'width': 2.5, 'color': color},
                    'hovertemplate': '<b>%{fullData.name}</b><br>' +
                                   'Дата: %{x}<br>' +
                                   'Выручка: %{y:,.0f} ₽<br>' +
                                   '<extra></extra>'
                })
            
            # Прогноз
            forecast_data = cafe_data[cafe_data['revenue_forecast'].notna()]
            if not forecast_data.empty:
                traces.append({
                    'x': forecast_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'y': forecast_data['revenue_forecast'].round(2).tolist(),
                    'name': f'{cafe} - Прогноз',
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {'width': 2.5, 'dash': 'dash', 'color': color},
                    'hovertemplate': '<b>%{fullData.name}</b><br>' +
                                   'Дата: %{x}<br>' +
                                   'Прогноз: %{y:,.0f} ₽<br>' +
                                   '<extra></extra>'
                })
                
                # Доверительный интервал
                x_values = forecast_data['ds'].dt.strftime('%Y-%m-%d').tolist()
                y_upper = forecast_data['revenue_upper'].round(2).tolist()
                y_lower = forecast_data['revenue_lower'].round(2).tolist()
                
                x_rev = x_values[::-1]
                y_lower_rev = y_lower[::-1]
                
                traces.append({
                    'x': x_values + x_rev,
                    'y': y_upper + y_lower_rev,
                    'fill': 'toself',
                    'fillcolor': f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    'line': {'color': 'transparent'},
                    'showlegend': False,
                    'name': f'{cafe} - Интервал',
                    'type': 'scatter',
                    'hoverinfo': 'skip'
                })
        
        return traces
    
    def _prepare_plot_layout(self) -> dict:
        """Подготовка layout для графика Plotly"""
        return {
            'title': {
                'text': 'Прогноз выручки сети "Пироговый Дворик"',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            'xaxis': {
                'title': 'Дата',
                'showgrid': True,
                'gridcolor': 'rgba(128, 128, 128, 0.2)',
                'showline': True,
                'linewidth': 1,
                'linecolor': 'rgba(128, 128, 128, 0.4)'
            },
            'yaxis': {
                'title': 'Выручка, ₽',
                'tickformat': ',.0f',
                'showgrid': True,
                'gridcolor': 'rgba(128, 128, 128, 0.2)',
                'showline': True,
                'linewidth': 1,
                'linecolor': 'rgba(128, 128, 128, 0.4)'
            },
            'hovermode': 'x unified',
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5,
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': 'rgba(128, 128, 128, 0.2)',
                'borderwidth': 1
            },
            'margin': {'l': 80, 'r': 50, 't': 80, 'b': 100},
            'plot_bgcolor': 'white',
            'paper_bgcolor': '#f8f9fa',
            'font': {'family': 'Arial, sans-serif', 'size': 12}
        }
    
    def _prepare_summary_data(self, df: pd.DataFrame) -> dict:
        """Подготовка сводных данных для таблицы"""
        summary_data = {
            'cafes': {},
            'total': {
                'revenue': {'actual': 0, 'forecast': 0, 'total': 0},
                'traffic': {'actual': 0, 'forecast': 0, 'total': 0}
            }
        }
        
        if df.empty:
            return summary_data
        
        today = pd.to_datetime(datetime.now().date())
        
        for cafe in df['cafe'].unique():
            cafe_df = df[df['cafe'] == cafe]
            
            # Актуальные данные
            actual_df = cafe_df[cafe_df['ds'] <= today]
            revenue_actual = actual_df['revenue_fact'].fillna(0).sum()
            traffic_actual = actual_df['traffic_fact'].fillna(0).sum()
            
            # Прогнозные данные
            forecast_df = cafe_df[cafe_df['ds'] > today]
            revenue_forecast = forecast_df['revenue_forecast'].fillna(0).sum()
            traffic_forecast = forecast_df['traffic_forecast'].fillna(0).sum()
            
            # Общие данные
            revenue_total = revenue_actual + revenue_forecast
            traffic_total = traffic_actual + traffic_forecast
            
            summary_data['cafes'][cafe] = {
                'revenue': {
                    'actual': revenue_actual,
                    'forecast': revenue_forecast,
                    'total': revenue_total
                },
                'traffic': {
                    'actual': traffic_actual,
                    'forecast': traffic_forecast,
                    'total': traffic_total
                }
            }
            
            # Добавляем к общим итогам
            summary_data['total']['revenue']['actual'] += revenue_actual
            summary_data['total']['revenue']['forecast'] += revenue_forecast
            summary_data['total']['revenue']['total'] += revenue_total
            summary_data['total']['traffic']['actual'] += traffic_actual
            summary_data['total']['traffic']['forecast'] += traffic_forecast
            summary_data['total']['traffic']['total'] += traffic_total
        
        return summary_data
    
    def export_excel(self):
        """Экспорт результатов в Excel"""
        try:
            if 'latest' not in self.forecast_cache:
                return jsonify({'error': 'Нет данных для экспорта'}), 404
            
            df = self.forecast_cache['latest']
            
            # Подготовка данных для экспорта
            export_df = self._prepare_export_data(df)
            
            # Создание временного файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f'forecast_{timestamp}.xlsx')
            
            # Сохранение в Excel с форматированием
            with pd.ExcelWriter(temp_file, engine='xlsxwriter') as writer:
                export_df.to_excel(writer, sheet_name='Прогноз', index=False)
                
                # Добавляем лист с метриками
                if self.metrics_cache:
                    metrics_df = self._prepare_metrics_dataframe()
                    metrics_df.to_excel(writer, sheet_name='Метрики качества', index=False)
                
                # Форматирование
                workbook = writer.book
                worksheet = writer.sheets['Прогноз']
                
                # Форматы
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D3D3D3',
                    'border': 1
                })
                
                date_format = workbook.add_format({'num_format': 'dd.mm.yyyy'})
                num_format = workbook.add_format({'num_format': '#,##0'})
                money_format = workbook.add_format({'num_format': '#,##0.00 ₽'})
                
                # Применяем форматы
                for col_num, value in enumerate(export_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Устанавливаем ширину колонок и форматы
                for idx, col in enumerate(export_df.columns):
                    if col == 'Дата':
                        worksheet.set_column(idx, idx, 12, date_format)
                    elif col == 'Кафе':
                        worksheet.set_column(idx, idx, 15)
                    elif 'Трафик' in col:
                        worksheet.set_column(idx, idx, 15, num_format)
                    else:
                        worksheet.set_column(idx, idx, 18, money_format)
            
            # Отправка файла
            response = send_file(
                temp_file,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'forecast_{timestamp}.xlsx'
            )
            
            # Удаление временного файла после отправки
            @response.call_on_close
            def remove_file():
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _prepare_export_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка данных для экспорта"""
        export_df = df.copy()
        
        # Переименовываем колонки
        rename_map = {
            'ds': 'Дата',
            'cafe': 'Кафе',
            'traffic_fact': 'Трафик_факт',
            'traffic_forecast': 'Трафик_прогноз',
            'check_fact': 'Средний_чек_факт',
            'check_forecast': 'Средний_чек_прогноз',
            'revenue_fact': 'Выручка_факт',
            'revenue_forecast': 'Выручка_прогноз',
            'revenue_lower': 'Выручка_прогноз_мин',
            'revenue_upper': 'Выручка_прогноз_макс'
        }
        
        columns_to_rename = {old: new for old, new in rename_map.items() if old in export_df.columns}
        export_df = export_df.rename(columns=columns_to_rename)
        
        # Выбираем только нужные колонки
        desired_columns = list(rename_map.values())
        available_columns = [col for col in desired_columns if col in export_df.columns]
        
        return export_df[available_columns]
    
    def _prepare_metrics_dataframe(self) -> pd.DataFrame:
        """Подготовка DataFrame с метриками"""
        metrics_data = []
        
        for cafe, metrics in self.metrics_cache.items():
            for metric_type in ['revenue', 'traffic', 'check']:
                row = {
                    'Кафе': cafe,
                    'Показатель': {
                        'revenue': 'Выручка',
                        'traffic': 'Трафик',
                        'check': 'Средний чек'
                    }[metric_type]
                }
                
                for metric_name in ['MAPE', 'MAE', 'RMSE', 'R2']:
                    key = f'{metric_name}_{metric_type}'
                    if key in metrics:
                        row[metric_name] = metrics[key]
                    else:
                        row[metric_name] = None
                
                metrics_data.append(row)
        
        return pd.DataFrame(metrics_data)
    
    def get_recommendations(self):
        """Получение рекомендаций для улучшения прогноза"""
        try:
            data = request.json
            cafe = data.get('cafe')
            model_type = data.get('model_type', 'prophet')
            
            if not cafe or cafe not in self.metrics_cache:
                return jsonify({'error': 'Нет метрик для данного кафе'}), 404
            
            metrics = self.metrics_cache[cafe]
            
            # Получаем рекомендации
            from utils.recommendations import RecommendationEngine
            rec_engine = RecommendationEngine(self.config)
            recommendations = rec_engine.get_recommendations_for_cafe(cafe, metrics, model_type)
            
            return jsonify({'recommendations': recommendations})
            
        except Exception as e:
            logger.error(f"Ошибка при получении рекомендаций: {e}")
            return jsonify({'error': str(e)}), 500
    
    def list_presets(self):
        """Получение списка пресетов"""
        try:
            presets = self.preset_manager.list_presets()
            return jsonify({'presets': presets})
        except Exception as e:
            logger.error(f"Ошибка при получении списка пресетов: {e}")
            return jsonify({'error': str(e)}), 500
    
    def load_preset(self, preset_id: int):
        """Загрузка пресета"""
        try:
            preset = self.preset_manager.load_preset(preset_id)
            if preset:
                return jsonify({'preset': preset})
            else:
                return jsonify({'error': 'Пресет не найден'}), 404
        except Exception as e:
            logger.error(f"Ошибка при загрузке пресета: {e}")
            return jsonify({'error': str(e)}), 500
    
    def save_preset(self):
        """Сохранение пресета"""
        try:
            data = request.json
            preset_id = self.preset_manager.save_preset(
                name=data.get('name'),
                cafes=data.get('cafes', []),
                params=data.get('params', {}),
                adjustments=data.get('adjustments', [])
            )
            return jsonify({'preset_id': preset_id})
        except Exception as e:
            logger.error(f"Ошибка при сохранении пресета: {e}")
            return jsonify({'error': str(e)}), 500
    
    def delete_preset(self, preset_id: int):
        """Удаление пресета"""
        try:
            success = self.preset_manager.delete_preset(preset_id)
            if success:
                return jsonify({'status': 'success'})
            else:
                return jsonify({'error': 'Пресет не найден'}), 404
        except Exception as e:
            logger.error(f"Ошибка при удалении пресета: {e}")
            return jsonify({'error': str(e)}), 500
    
    def list_adjustments(self):
        """Получение списка корректировок"""
        try:
            adjustments = self.preset_manager.list_adjustments()
            return jsonify({'adjustments': adjustments})
        except Exception as e:
            logger.error(f"Ошибка при получении списка корректировок: {e}")
            return jsonify({'error': str(e)}), 500
    
    def save_adjustment(self):
        """Сохранение корректировки"""
        try:
            data = request.json
            adjustment_id = self.preset_manager.save_adjustment(
                cafe=data.get('cafe'),
                date_from=data.get('date_from'),
                date_to=data.get('date_to'),
                metric=data.get('metric'),
                coefficient=data.get('coefficient'),
                reason=data.get('reason')
            )
            return jsonify({'adjustment_id': adjustment_id})
        except Exception as e:
            logger.error(f"Ошибка при сохранении корректировки: {e}")
            return jsonify({'error': str(e)}), 500
    
    def delete_adjustment(self, adjustment_id: int):
        """Удаление корректировки"""
        try:
            success = self.preset_manager.delete_adjustment(adjustment_id)
            if success:
                return jsonify({'status': 'success'})
            else:
                return jsonify({'error': 'Корректировка не найдена'}), 404
        except Exception as e:
            logger.error(f"Ошибка при удалении корректировки: {e}")
            return jsonify({'error': str(e)}), 500