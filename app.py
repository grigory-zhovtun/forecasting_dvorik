"""
Веб-приложение для прогнозирования выручки сети кафе "Пироговый Дворик"
"""

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import json
from datetime import datetime, timedelta
import os
import warnings
import logging
from pathlib import Path
import plotly.graph_objs as go
import plotly.utils
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
import tempfile

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

app = Flask(__name__)

# Глобальные переменные для хранения данных
data_cache = {}
forecast_cache = {}
progress_queue = queue.Queue()
current_task = None


class ForecastEngine:
    def __init__(self):
        self.data_path = r'C:\Users\zhovtun.ga\Documents\automate\check_and_trafic\facts.xlsx'
        self.holidays_path = r'C:\Users\zhovtun.ga\Documents\Notebooks\planing-revenue\holidays_df.xlsx'
        self.load_data()

    def load_data(self):
        """Загрузка данных"""
        try:
            # Загрузка фактических данных
            self.df = pd.read_excel(self.data_path)
            self.df = self.df[~self.df['Кафе'].isin(['ПД-15', 'ПД-20', 'ПД-25', 'Кейтеринг'])]
            self.df = self.df.dropna(subset=['Дата', 'Тр', 'Чек'])
            self.df['Дата'] = pd.to_datetime(self.df['Дата'])

            # Расчет выручки
            self.df['Выручка'] = self.df['Тр'] * self.df['Чек']

            # Загрузка праздников
            self.load_holidays()

            # Сохранение в кэш
            data_cache['df'] = self.df
            data_cache['cafes'] = sorted(self.df['Кафе'].unique().tolist())
            data_cache['date_range'] = {
                'min': self.df['Дата'].min().strftime('%Y-%m-%d'),
                'max': self.df['Дата'].max().strftime('%Y-%m-%d')
            }

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")

    def load_holidays(self):
        """Загрузка праздников"""
        try:
            custom_holidays = pd.read_excel(self.holidays_path)
        except:
            custom_holidays = pd.DataFrame(columns=['ds', 'holiday'])

        years = list(range(2020, 2027))
        ru_holidays = make_holidays_df(year_list=years, country='RU')
        self.holidays = pd.concat([custom_holidays, ru_holidays]).drop_duplicates().reset_index(drop=True)

    def classify_cafe(self, cafe):
        """Классификация кафе"""
        df_cafe = self.df[self.df['Кафе'] == cafe]
        unique_days = df_cafe['Дата'].nunique()

        if unique_days >= 365:
            return "mature"
        elif unique_days >= 180:
            return "established"
        elif unique_days >= 90:
            return "developing"
        else:
            return "new"

    def remove_outliers(self, df_cafe, column, multiplier=1.5):
        """Удаление выбросов"""
        Q1 = df_cafe[column].quantile(0.25)
        Q3 = df_cafe[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        return df_cafe[(df_cafe[column] >= lower_bound) & (df_cafe[column] <= upper_bound)]

    def forecast_cafe(self, cafe, params):
        """Прогнозирование для одного кафе"""
        try:
            # Фильтрация данных
            df_cafe = self.df[self.df['Кафе'] == cafe].copy()
            df_cafe = df_cafe.groupby('Дата').agg({
                'Тр': 'sum',
                'Чек': 'mean',
                'Выручка': 'sum'
            }).reset_index()

            # Удаление выбросов если включено
            if params.get('remove_outliers', True):
                multiplier = params.get('outlier_multiplier', 1.5)
                df_cafe = self.remove_outliers(df_cafe, 'Тр', multiplier)
                df_cafe = self.remove_outliers(df_cafe, 'Чек', multiplier)

            # Заполнение пропусков
            df_cafe = df_cafe.set_index('Дата').asfreq('D').reset_index()
            df_cafe['Тр'] = df_cafe['Тр'].fillna(df_cafe['Тр'].mean())
            df_cafe['Чек'] = df_cafe['Чек'].fillna(df_cafe['Чек'].mean())
            df_cafe['Выручка'] = df_cafe['Тр'] * df_cafe['Чек']

            # Прогнозирование трафика
            df_traffic = df_cafe[['Дата', 'Тр']].rename(columns={'Дата': 'ds', 'Тр': 'y'})

            m_traffic = Prophet(
                holidays=self.holidays,
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                yearly_seasonality=params.get('yearly_seasonality', True),
                weekly_seasonality=params.get('weekly_seasonality', True),
                daily_seasonality=params.get('daily_seasonality', False)
            )

            m_traffic.fit(df_traffic)

            future_traffic = m_traffic.make_future_dataframe(
                periods=params.get('forecast_horizon', 365)
            )
            forecast_traffic = m_traffic.predict(future_traffic)

            # Прогнозирование среднего чека
            df_check = df_cafe[['Дата', 'Чек']].rename(columns={'Дата': 'ds', 'Чек': 'y'})

            m_check = Prophet(
                holidays=self.holidays,
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                yearly_seasonality=params.get('yearly_seasonality', True),
                weekly_seasonality=params.get('weekly_seasonality', True),
                daily_seasonality=False
            )

            m_check.fit(df_check)

            future_check = m_check.make_future_dataframe(
                periods=params.get('forecast_horizon', 365)
            )
            forecast_check = m_check.predict(future_check)

            # Объединение результатов
            result = pd.DataFrame({
                'ds': forecast_traffic['ds'],
                'traffic_forecast': forecast_traffic['yhat'],
                'traffic_lower': forecast_traffic['yhat_lower'],
                'traffic_upper': forecast_traffic['yhat_upper'],
                'check_forecast': forecast_check['yhat'],
                'check_lower': forecast_check['yhat_lower'],
                'check_upper': forecast_check['yhat_upper']
            })

            # Расчет прогноза выручки
            result['revenue_forecast'] = result['traffic_forecast'] * result['check_forecast']
            result['revenue_lower'] = result['traffic_lower'] * result['check_lower']
            result['revenue_upper'] = result['traffic_upper'] * result['check_upper']

            # Добавление фактических данных
            result = result.merge(
                df_cafe[['Дата', 'Тр', 'Чек', 'Выручка']].rename(columns={
                    'Дата': 'ds',
                    'Тр': 'traffic_fact',
                    'Чек': 'check_fact',
                    'Выручка': 'revenue_fact'
                }),
                on='ds',
                how='left'
            )

            result['cafe'] = cafe

            # Применение корректировок
            if 'adjustments' in params and params['adjustments']:
                for adj in params['adjustments']:
                    if adj['cafe'] == cafe or adj['cafe'] == 'ALL':
                        date_from = pd.to_datetime(adj['date_from'])
                        date_to = pd.to_datetime(adj['date_to'])
                        mask = (result['ds'] >= date_from) & (result['ds'] <= date_to)

                        if adj['metric'] in ['traffic', 'both']:
                            result.loc[mask, 'traffic_forecast'] *= adj['coefficient']

                        if adj['metric'] in ['check', 'both']:
                            result.loc[mask, 'check_forecast'] *= adj['coefficient']

                        if adj['metric'] == 'both':
                            result.loc[mask, 'revenue_forecast'] *= adj['coefficient']

            return result

        except Exception as e:
            print(f"Ошибка прогнозирования для {cafe}: {e}")
            return None

    def forecast_all(self, cafes, params, progress_callback=None):
        """Прогнозирование для всех выбранных кафе"""
        results = []
        total = len(cafes)

        for i, cafe in enumerate(cafes):
            if progress_callback:
                progress_callback(i + 1, total, f"Обработка {cafe}")

            result = self.forecast_cafe(cafe, params)
            if result is not None:
                results.append(result)

        if results:
            combined = pd.concat(results, ignore_index=True)
            return combined
        else:
            return pd.DataFrame()

    def prepare_export_data(self, forecast_df):
        """Подготовка данных для экспорта в Excel"""
        # Создаем копию для безопасности
        export_df = forecast_df.copy()

        # Проверяем какие колонки есть в датафрейме
        print("Доступные колонки в forecast_df:")
        print(list(export_df.columns))

        # Переименовываем существующие колонки
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

        # Переименовываем только существующие колонки
        columns_to_rename = {}
        for old_col, new_col in rename_map.items():
            if old_col in export_df.columns:
                columns_to_rename[old_col] = new_col

        export_df = export_df.rename(columns=columns_to_rename)

        # Список желаемых колонок для экспорта
        desired_columns = [
            'Дата', 'Кафе',
            'Трафик_факт', 'Трафик_прогноз',
            'Средний_чек_факт', 'Средний_чек_прогноз',
            'Выручка_факт', 'Выручка_прогноз',
            'Выручка_прогноз_мин', 'Выручка_прогноз_макс'
        ]

        # Выбираем только существующие колонки
        available_columns = [col for col in desired_columns if col in export_df.columns]

        print(f"Колонки для экспорта: {available_columns}")

        return export_df[available_columns]


# Создание экземпляра движка
engine = ForecastEngine()


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/get_initial_data')
def get_initial_data():
    """Получение начальных данных"""
    return jsonify({
        'cafes': data_cache.get('cafes', []),
        'date_range': data_cache.get('date_range', {}),
        'default_params': {
            'forecast_horizon': 365,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'outlier_multiplier': 1.5,
            'remove_outliers': True,
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
    })


@app.route('/api/forecast', methods=['POST'])
def forecast():
    """Запуск прогнозирования"""
    global current_task

    params = request.json
    selected_cafes = params.get('cafes', [])

    if not selected_cafes:
        return jsonify({'error': 'Не выбрано ни одного кафе'}), 400

    # Запуск прогнозирования в отдельном потоке
    def run_forecast():
        def progress_callback(current, total, message):
            progress_queue.put({
                'current': current,
                'total': total,
                'message': message,
                'percentage': int((current / total) * 100)
            })

        result = engine.forecast_all(selected_cafes, params, progress_callback)
        forecast_cache['latest'] = result
        progress_queue.put({'status': 'complete'})

    current_task = threading.Thread(target=run_forecast)
    current_task.start()

    return jsonify({'status': 'started'})


@app.route('/api/progress')
def get_progress():
    """Получение прогресса выполнения"""
    try:
        progress = progress_queue.get_nowait()
        return jsonify(progress)
    except queue.Empty:
        return jsonify({'status': 'waiting'})


@app.route('/api/get_forecast_data')
def get_forecast_data():
    """Получение данных прогноза для визуализации"""
    try:
        if 'latest' not in forecast_cache:
            return jsonify({'error': 'Нет данных прогноза'}), 404

        df = forecast_cache['latest'].copy()

        # Проверка наличия данных
        if df.empty:
            return jsonify({
                'data': [],
                'layout': {'title': 'Нет данных для отображения'}
            })

        # Фильтрация по датам если указаны
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')

        if date_from:
            df = df[df['ds'] >= pd.to_datetime(date_from)]
        if date_to:
            df = df[df['ds'] <= pd.to_datetime(date_to)]

        # Преобразование в формат для Plotly
        traces = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for idx, cafe in enumerate(df['cafe'].unique()):
            cafe_data = df[df['cafe'] == cafe].sort_values('ds')
            color = colors[idx % len(colors)]

            # Факт - только там где есть данные
            fact_data = cafe_data[cafe_data['revenue_fact'].notna()]
            if not fact_data.empty:
                traces.append({
                    'x': fact_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'y': fact_data['revenue_fact'].round(2).tolist(),
                    'name': f'{cafe} - Факт',
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {
                        'width': 2.5,
                        'color': color
                    },
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
                    'line': {
                        'width': 2.5,
                        'dash': 'dash',
                        'color': color
                    },
                    'hovertemplate': '<b>%{fullData.name}</b><br>' +
                                     'Дата: %{x}<br>' +
                                     'Прогноз: %{y:,.0f} ₽<br>' +
                                     '<extra></extra>'
                })

                # Доверительный интервал
                x_values = forecast_data['ds'].dt.strftime('%Y-%m-%d').tolist()
                y_upper = forecast_data['revenue_upper'].round(2).tolist()
                y_lower = forecast_data['revenue_lower'].round(2).tolist()

                # Создаем замкнутый полигон для заливки
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

        # Компоновка графика
        layout = {
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
            'font': {
                'family': 'Arial, sans-serif',
                'size': 12
            }
        }

        return jsonify({
            'data': traces,
            'layout': layout
        })

    except Exception as e:
        app.logger.error(f"Ошибка при подготовке данных для графика: {str(e)}")
        return jsonify({
            'error': f'Ошибка при подготовке данных: {str(e)}'
        }), 500


@app.route('/api/export_excel')
def export_excel():
    """Экспорт результатов в Excel"""
    try:
        if 'latest' not in forecast_cache:
            return jsonify({'error': 'Нет данных для экспорта'}), 404

        df = forecast_cache['latest']

        # Отладочная информация
        print(f"Тип данных в кэше: {type(df)}")
        print(f"Размер данных: {df.shape if hasattr(df, 'shape') else 'N/A'}")

        export_df = engine.prepare_export_data(df)

        # Создание временного файла с уникальным именем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f'forecast_{timestamp}.xlsx')

        # Сохранение в Excel
        with pd.ExcelWriter(temp_file, engine='xlsxwriter') as writer:
            export_df.to_excel(writer, sheet_name='Прогноз', index=False)

            # Получаем объекты для форматирования
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

            # Применяем форматы к заголовкам
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
                else:  # Денежные колонки
                    worksheet.set_column(idx, idx, 18, money_format)

        # Отправка файла
        response = send_file(
            temp_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'forecast_{timestamp}.xlsx'
        )

        # Удаление временного файла после отправки
        def remove_file(response):
            try:
                os.remove(temp_file)
            except:
                pass
            return response

        response.call_on_close(lambda: remove_file(response))

        return response

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        app.logger.error(f"Ошибка при экспорте: {str(e)}\n{error_details}")
        return jsonify({
            'error': f'Ошибка при экспорте: {str(e)}',
            'details': error_details
        }), 500


if __name__ == '__main__':
    # Создание папки для шаблонов если не существует
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    app.run(debug=True, port=5000)