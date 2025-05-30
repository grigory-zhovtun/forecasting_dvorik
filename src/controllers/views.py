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
    
    def __init__(self, app, config: dict, data_loader, forecast_engine):
        """
        Инициализация представлений
        
        Args:
            app: Flask приложение
            config: Конфигурация
            data_loader: Экземпляр DataLoader
            forecast_engine: Экземпляр ForecastEngine
        """
        self.app = app
        self.config = config
        self.data_loader = data_loader
        self.forecast_engine = forecast_engine
        
        # Кэши и очереди
        self.forecast_cache = {}
        self.metrics_cache = {}
        self.progress_queue = queue.Queue()
        self.current_task = None
        self.cancel_flag = threading.Event()
        
        # Добавляем хранилища для результатов
        self.app.forecast_results = {}
        self.app.adjusted_forecast_results = {}
        
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
        
        
        # Маршруты для корректировок
        self.app.route('/api/adjustments/list')(self.list_adjustments)
        self.app.route('/api/adjustments/save', methods=['POST'])(self.save_adjustment)
        self.app.route('/api/adjustments/delete/<int:adjustment_id>', methods=['DELETE'])(self.delete_adjustment)
        self.app.route('/api/apply_adjustments', methods=['POST'])(self.apply_adjustments)
        
        # Маршрут для паспорта
        self.app.route('/api/passport_data')(self.get_passport_data)
    
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
                # Сохраняем результаты для корректировок
                session_id = 'default'  # Используем дефолтный session_id
                self.app.forecast_results[session_id] = result
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
                from src.utils.recommendations import RecommendationEngine
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
        """Подготовка данных для графика Plotly с агрегированием по всем кафе"""
        traces = []
        
        # Агрегируем данные по всем кафе
        df_agg = df.groupby('ds').agg({
            'revenue_fact': 'sum',
            'revenue_forecast': 'sum',
            'revenue_upper': 'sum',
            'revenue_lower': 'sum'
        }).reset_index()
        
        # Факт (агрегированный)
        fact_data = df_agg[df_agg['revenue_fact'].notna()].sort_values('ds')
        if not fact_data.empty:
            traces.append({
                'x': fact_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'y': fact_data['revenue_fact'].round(2).tolist(),
                'name': 'Факт (все кафе)',
                'type': 'scatter',
                'mode': 'lines',
                'line': {'width': 3, 'color': '#1f77b4'},
                'hovertemplate': '<b>%{fullData.name}</b><br>' +
                               'Дата: %{x}<br>' +
                               'Выручка: %{y:,.0f} ₽<br>' +
                               '<extra></extra>'
            })
            
            # Добавляем линию тренда для фактических данных
            try:
                from sklearn.linear_model import LinearRegression
                import numpy as np
                
                # Преобразуем даты в числовой формат для регрессии
                x_numeric = np.arange(len(fact_data)).reshape(-1, 1)
                y_values = fact_data['revenue_fact'].values
                
                # Обучаем модель линейной регрессии
                lr = LinearRegression()
                lr.fit(x_numeric, y_values)
                trend_values = lr.predict(x_numeric)
                
                traces.append({
                    'x': fact_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'y': trend_values.round(2).tolist(),
                    'name': 'Тренд',
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {'width': 2, 'dash': 'dot', 'color': '#ff7f0e'},
                    'hovertemplate': '<b>%{fullData.name}</b><br>' +
                                   'Дата: %{x}<br>' +
                                   'Тренд: %{y:,.0f} ₽<br>' +
                                   '<extra></extra>'
                })
            except Exception as e:
                logger.warning(f"Не удалось добавить линию тренда: {e}")
        
        # Прогноз (агрегированный)
        forecast_data = df_agg[df_agg['revenue_forecast'].notna()].sort_values('ds')
        if not forecast_data.empty:
            traces.append({
                'x': forecast_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'y': forecast_data['revenue_forecast'].round(2).tolist(),
                'name': 'Прогноз (все кафе)',
                'type': 'scatter',
                'mode': 'lines',
                'line': {'width': 3, 'dash': 'dash', 'color': '#2ca02c'},
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
                'fillcolor': 'rgba(44, 160, 44, 0.2)',
                'line': {'color': 'transparent'},
                'showlegend': False,
                'name': 'Доверительный интервал',
                'type': 'scatter',
                'hoverinfo': 'skip'
            })
        
        return traces
    
    def apply_adjustments(self):
        """Применение корректировок к данным прогноза"""
        try:
            data = request.get_json()
            adjustments = data.get('adjustments', [])
            
            # Получаем текущие данные прогноза
            session_id = 'default'  # Используем дефолтный session_id
            if session_id not in self.app.forecast_results:
                # Пробуем получить из кэша
                if 'latest' in self.forecast_cache:
                    self.app.forecast_results[session_id] = self.forecast_cache['latest']
                else:
                    return jsonify({'error': 'Нет данных прогноза. Сначала выполните прогноз.'}), 404
            
            # Копируем исходные данные
            original_df = self.app.forecast_results[session_id].copy()
            adjusted_df = original_df.copy()
            
            # Применяем корректировки
            for adj in adjustments:
                cafe = adj.get('cafe')
                metric = adj.get('metric')
                adj_type = adj.get('type')
                value = adj.get('value')
                date_from = pd.to_datetime(adj.get('dateFrom'))
                date_to = pd.to_datetime(adj.get('dateTo'))
                
                # Фильтр по кафе
                if cafe == 'ALL':
                    mask = (adjusted_df['ds'] >= date_from) & (adjusted_df['ds'] <= date_to)
                else:
                    mask = (adjusted_df['cafe'] == cafe) & (adjusted_df['ds'] >= date_from) & (adjusted_df['ds'] <= date_to)
                
                # Применяем корректировку
                if metric == 'revenue':
                    if adj_type == 'percent':
                        adjusted_df.loc[mask, 'revenue_forecast'] *= (1 + value / 100)
                    else:
                        adjusted_df.loc[mask, 'revenue_forecast'] += value
                elif metric == 'traffic':
                    if adj_type == 'percent':
                        adjusted_df.loc[mask, 'traffic_forecast'] *= (1 + value / 100)
                    else:
                        adjusted_df.loc[mask, 'traffic_forecast'] += value
                elif metric == 'avg_check':
                    if adj_type == 'percent':
                        adjusted_df.loc[mask, 'avg_check_forecast'] *= (1 + value / 100)
                    else:
                        adjusted_df.loc[mask, 'avg_check_forecast'] += value
                
                # Пересчитываем выручку если изменили трафик или чек
                if metric in ['traffic', 'avg_check']:
                    adjusted_df.loc[mask, 'revenue_forecast'] = (
                        adjusted_df.loc[mask, 'traffic_forecast'] * 
                        adjusted_df.loc[mask, 'avg_check_forecast']
                    )
            
            # Сохраняем скорректированные данные
            self.app.adjusted_forecast_results[session_id] = adjusted_df
            
            # Подготавливаем данные для графика
            traces = self._prepare_plot_traces(adjusted_df)
            layout = self._prepare_plot_layout()
            
            # Подготавливаем сводные данные
            summary_data = self._prepare_summary_data(adjusted_df)
            
            return jsonify({
                'data': traces,
                'layout': layout,
                'summary_data': summary_data
            })
            
        except Exception as e:
            logger.error(f"Ошибка при применении корректировок: {e}")
            return jsonify({'error': str(e)}), 500
    
    def _prepare_summary_data(self, df: pd.DataFrame) -> dict:
        """Подготовка сводных данных для таблицы"""
        summary = {'cafes': {}, 'total': {}}
        
        for cafe in df['cafe'].unique():
            cafe_data = df[df['cafe'] == cafe]
            
            # Считаем суммы по факту и прогнозу
            fact_revenue = cafe_data['revenue_fact'].sum()
            forecast_revenue = cafe_data['revenue_forecast'].sum()
            fact_traffic = cafe_data['traffic_fact'].sum()
            forecast_traffic = cafe_data['traffic_forecast'].sum()
            
            # Средний чек
            fact_avg_check = fact_revenue / fact_traffic if fact_traffic > 0 else 0
            forecast_avg_check = forecast_revenue / forecast_traffic if forecast_traffic > 0 else 0
            
            summary['cafes'][cafe] = {
                'revenue': {
                    'actual': fact_revenue,
                    'forecast': forecast_revenue,
                    'total': fact_revenue + forecast_revenue
                },
                'traffic': {
                    'actual': fact_traffic,
                    'forecast': forecast_traffic,
                    'total': fact_traffic + forecast_traffic
                },
                'avg_check': {
                    'actual': fact_avg_check,
                    'forecast': forecast_avg_check,
                    'total': (fact_avg_check + forecast_avg_check) / 2
                }
            }
        
        # Общие итоги
        total_fact_revenue = df['revenue_fact'].sum()
        total_forecast_revenue = df['revenue_forecast'].sum()
        total_fact_traffic = df['traffic_fact'].sum()
        total_forecast_traffic = df['traffic_forecast'].sum()
        
        summary['total'] = {
            'revenue': {
                'actual': total_fact_revenue,
                'forecast': total_forecast_revenue,
                'total': total_fact_revenue + total_forecast_revenue
            },
            'traffic': {
                'actual': total_fact_traffic,
                'forecast': total_forecast_traffic,
                'total': total_fact_traffic + total_forecast_traffic
            }
        }
        
        return summary
    
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
            from src.utils.recommendations import RecommendationEngine
            rec_engine = RecommendationEngine(self.config)
            recommendations = rec_engine.get_recommendations_for_cafe(cafe, metrics, model_type)
            
            return jsonify({'recommendations': recommendations})
            
        except Exception as e:
            logger.error(f"Ошибка при получении рекомендаций: {e}")
            return jsonify({'error': str(e)}), 500
    
    def list_adjustments(self):
        """Получение списка корректировок"""
        # Временная заглушка
        return jsonify({'adjustments': []})
    
    def save_adjustment(self):
        """Сохранение корректировки"""
        # Временная заглушка
        return jsonify({'adjustment_id': 1})
    
    def delete_adjustment(self, adjustment_id: int):
        """Удаление корректировки"""
        # Временная заглушка
        return jsonify({'status': 'success'})
    
    def get_passport_data(self):
        """Получение данных паспорта для всех кафе"""
        try:
            # Загружаем данные паспорта
            passport_df = self.data_loader.load_passport_data()

            # Logging for passport_df
            logger.info(f"Passport data received in views - shape: {passport_df.shape}")
            logger.info(f"Passport data received in views - columns: {passport_df.columns.tolist()}")
            logger.info(f"Passport data received in views - head:\n{passport_df.head().to_string()}")
            
            if passport_df.empty:
                logger.warning("Данные паспорта пустые")
                return jsonify({'passport_data': []})
            
            # Получаем факты для вычисления динамических показателей
            facts_df = self.data_loader.load_facts_data()
            
            passport_data = []
            
            for _, row in passport_df.iterrows():
                cafe_name = row['cafe']
                cafe_facts = facts_df[facts_df['Кафе'] == cafe_name]
                
                # Вычисляем дни работы
                if not cafe_facts.empty:
                    open_date = pd.to_datetime(row['open_date'])
                    today = datetime.now()
                    days_working = (today - open_date).days
                    
                    # Средние показатели за последние 30 дней
                    last_30_days = cafe_facts[cafe_facts['Дата'] >= (today - timedelta(days=30))]
                    
                    avg_traffic = 0  # Initialize
                    avg_revenue = 0  # Initialize
                    
                    if not last_30_days.empty:
                        if 'Тр' in last_30_days.columns:
                            avg_traffic = last_30_days['Тр'].mean()
                        else:
                            logger.warning(f"Column 'Тр' not found in last_30_days for cafe: {cafe_name}. Setting avg_traffic to 0.")
                        
                        if 'Выручка' in last_30_days.columns:
                            avg_revenue = last_30_days['Выручка'].mean()
                        else:
                            logger.warning(f"Column 'Выручка' not found in last_30_days for cafe: {cafe_name}. Setting avg_revenue to 0.")
                    else:
                        logger.info(f"No data in last_30_days for cafe: {cafe_name}. Traffic and revenue metrics will be zero.")

                    # Чеки на сотрудника и ТО на сотрудника
                    checks_per_staff = avg_traffic / row['staff_count'] if row['staff_count'] > 0 else 0
                    revenue_per_staff = avg_revenue / row['staff_count'] if row['staff_count'] > 0 else 0
                    
                else:
                    days_working = 0
                    checks_per_staff = 0
                    revenue_per_staff = 0
                
                passport_item = {
                    'cafe': cafe_name,
                    'open_date': row['open_date'],
                    'days_working': days_working,
                    'seats': row['seats'],
                    'location_type': row['location_type'],
                    'staff_count': row['staff_count'],
                    'area_sqm': row['area_sqm'],
                    'parking': row['parking'],
                    'delivery': row['delivery'],
                    'checks_per_staff': round(checks_per_staff, 1),
                    'revenue_per_staff': round(revenue_per_staff, 0),
                    'kitchen_type': row.get('kitchen_type', 'Не указано'),
                    'work_hours': row.get('work_hours', 'Не указано')
                }
                
                passport_data.append(passport_item)

            # Logging for passport_data before returning
            logger.info(f"Processed passport_data to be returned - count: {len(passport_data)}")
            if passport_data:
                logger.info(f"First item of processed passport_data: {passport_data[0]}")
            else:
                logger.info("Processed passport_data is empty.")
            
            return jsonify({'passport_data': passport_data})
            
        except Exception as e:
            logger.error(f"Ошибка при получении данных паспорта: {e}")
            return jsonify({'error': str(e)}), 500