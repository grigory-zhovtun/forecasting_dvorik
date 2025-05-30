import requests
import pandas as pd
import holidays as pyholidays  # Используем alias во избежание конфликтов имен
from datetime import date, timedelta, datetime
import time

# --- Константы ---
# Минимальный год для сбора данных
START_YEAR = 2020
# Максимальный год (текущий + 2 для обеспечения прогноза на 2 года вперед)
CURRENT_YEAR = datetime.now().year
END_YEAR = max(CURRENT_YEAR + 2, 2026)  # Гарантируем, что календарь будет минимум до 2026 года

# URL API isdayoff.ru
ISDAYOFF_API_URL = "https://isdayoff.ru/api/getdata"


# --- Функции ---

def get_day_types_from_isdayoff(year):
    """
    Получает типы дней (0-рабочий, 1-выходной/праздник, 2-сокращенный) для года из API isdayoff.ru.
    Возвращает словарь {дата_строка: тип_дня_код} или None при ошибке.
    """
    params = {'year': year, 'pre': 1}  # pre=1 для учета сокращенных дней
    try:
        response = requests.get(ISDAYOFF_API_URL, params=params, timeout=15)  # Увеличен таймаут
        response.raise_for_status()
        day_types_str = response.text

        if not day_types_str or not all(c.isdigit() for c in day_types_str):
            print(f"⚠️ Ошибка: API isdayoff.ru вернул некорректные данные для года {year}: '{day_types_str[:100]}...'")
            return None

        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_year = 366 if is_leap else 365

        if len(day_types_str) != days_in_year:
            print(f"⚠️ Предупреждение: API isdayoff.ru для года {year} вернул строку длиной {len(day_types_str)}, "
                  f"ожидалось {days_in_year} дней. Данные за этот год могут быть неполными или неточными.")
            # Если критично, можно вернуть None. Пока продолжаем с тем, что есть, но с предупреждением.

        year_data = {}
        current_date_obj = date(year, 1, 1)
        for i, day_type_char in enumerate(day_types_str):
            if i >= days_in_year:  # Если строка от API длиннее, чем дней в году, обрезаем
                print(f"ℹ️ Информация: API isdayoff.ru для года {year} вернул строку длиннее ({len(day_types_str)}), "
                      f"чем дней в году ({days_in_year}). Используются только первые {days_in_year} значений.")
                break

            try:
                day_type_code = int(day_type_char)
                year_data[current_date_obj.strftime('%Y-%m-%d')] = day_type_code
            except ValueError:  # На случай, если проверка all(isdigit) пропустила что-то не то (маловероятно)
                print(
                    f"⚠️ Ошибка: Не удалось преобразовать символ '{day_type_char}' в число для даты {current_date_obj}. Пропуск.")
                year_data[current_date_obj.strftime('%Y-%m-%d')] = -1  # Помечаем как ошибку/неизвестно
            current_date_obj += timedelta(days=1)

        return year_data
    except requests.exceptions.Timeout:
        print(f"❌ Ошибка: Таймаут при запросе к isdayoff.ru для года {year}.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка: Проблема с запросом к isdayoff.ru для года {year}: {e}")
        return None


def get_official_holiday_names(start_year, end_year):
    """
    Получает официальные праздники России из файла prophet_holidays_2020_2027.csv.
    Возвращает словарь {дата_строка: название_праздника}.
    """
    try:
        # Используем предварительно созданный файл с праздниками
        holidays_file = 'prophet_holidays_2020_2027.csv'
        
        try:
            holidays_df = pd.read_csv(holidays_file)
            holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
            
            # Фильтруем по нужным годам и только официальные праздники (не предпраздничные)
            holidays_df = holidays_df[
                (holidays_df['ds'].dt.year >= start_year) & 
                (holidays_df['ds'].dt.year <= end_year) &
                (holidays_df['holiday'] != 'Предпраздничный')
            ]
            
            # Создаем словарь {дата_строка: название_праздника}
            holidays_dict = {row['ds'].strftime('%Y-%m-%d'): row['holiday'] 
                             for _, row in holidays_df.iterrows()}
            
            print(f"✅ Успешно загружены праздники РФ из файла {holidays_file}. Найдено {len(holidays_dict)} праздников.")
            return holidays_dict
            
        except Exception as file_error:
            print(f"⚠️ Ошибка при чтении файла с праздниками: {file_error}")
            # Создаем минимальный набор российских праздников вручную как запасной вариант
            holidays_dict = {}
            for year in range(start_year, end_year + 1):
                # Новый год
                holidays_dict[f"{year}-01-01"] = "Новый год"
                # Рождество
                holidays_dict[f"{year}-01-07"] = "Рождество Христово"
                # День защитника Отечества
                holidays_dict[f"{year}-02-23"] = "День защитника Отечества"
                # Международный женский день
                holidays_dict[f"{year}-03-08"] = "Международный женский день"
                # Праздник Весны и Труда
                holidays_dict[f"{year}-05-01"] = "Праздник Весны и Труда"
                # День Победы
                holidays_dict[f"{year}-05-09"] = "День Победы"
                # День России
                holidays_dict[f"{year}-06-12"] = "День России"
                # День народного единства
                holidays_dict[f"{year}-11-04"] = "День народного единства"
            print(f"ℹ️ Используем базовый набор российских праздников. Создано {len(holidays_dict)} записей.")
            return holidays_dict
            
    except Exception as e:
        print(f"❌ Ошибка при получении праздников: {e}")
        return {}


def create_full_calendar_dataframe(start_year, end_year):
    """
    Создает DataFrame с информацией о рабочих, выходных, праздничных
    и предпраздничных днях для указанного диапазона лет.
    """
    all_days_data = []

    print(f"🗓️ Получение названий официальных праздников РФ с {start_year} по {end_year}...")
    official_holiday_names = get_official_holiday_names(start_year, end_year)
    if not official_holiday_names:
        print("⚠️ Не удалось получить названия официальных праздников. Колонка с названиями будет пустой.")
    else:
        print(f"ℹ️ Получено {len(official_holiday_names)} официальных праздников для периода {start_year}-{end_year}.")

    isdayoff_calendar_data = {}
    for year_to_fetch in range(start_year, end_year + 1):
        print(f"⏳ Загрузка типов дней для {year_to_fetch} года из isdayoff.ru...")
        year_day_types = get_day_types_from_isdayoff(year_to_fetch)
        if year_day_types:
            isdayoff_calendar_data.update(year_day_types)
        else:
            print(
                f"❌ Не удалось загрузить данные о типах дней для {year_to_fetch} года. Этот год будет пропущен в данных isdayoff.")
        time.sleep(0.2)  # Небольшая задержка между запросами к API isdayoff.ru

    if not isdayoff_calendar_data:
        print("❌ Не удалось загрузить данные из isdayoff.ru ни для одного года. Невозможно создать полный календарь.")
        return pd.DataFrame()

    start_date_obj = date(start_year, 1, 1)
    end_date_obj = date(end_year, 12, 31)
    num_days = (end_date_obj - start_date_obj).days + 1

    for i in range(num_days):
        current_dt_obj = start_date_obj + timedelta(days=i)
        date_str = current_dt_obj.strftime('%Y-%m-%d')

        day_type_code_api = isdayoff_calendar_data.get(date_str,
                                                       -1)  # -1 если дата отсутствует (например, год не загрузился)

        holiday_name_pyholidays = official_holiday_names.get(date_str)
        is_official_holiday_flag = holiday_name_pyholidays is not None

        day_type_desc = 'unknown'
        is_day_off_flag = False
        is_short_day_flag = False

        if day_type_code_api == 0:
            day_type_desc = 'workday'
        elif day_type_code_api == 1:
            day_type_desc = 'day_off'  # Выходной или праздник по производственному календарю
            is_day_off_flag = True
        elif day_type_code_api == 2:
            day_type_desc = 'short_day'  # Сокращенный рабочий день
            is_short_day_flag = True
        elif day_type_code_api == -1:  # Данные от isdayoff отсутствуют для этой даты
            # Попробуем определить по дню недели и данным из pyholidays
            print(
                f"ℹ️ Информация: нет данных от isdayoff.ru для {date_str}. Тип дня определяется на основе дня недели/pyholidays.")
            weekday = current_dt_obj.weekday()  # Понедельник 0, Воскресенье 6
            if is_official_holiday_flag:
                day_type_desc = 'day_off'  # Официальный праздник считаем выходным
                is_day_off_flag = True
            elif weekday >= 5:  # Суббота или Воскресенье
                day_type_desc = 'day_off'  # Обычный выходной
                is_day_off_flag = True
            else:
                day_type_desc = 'workday'

        # Определение имени события для Prophet
        prophet_event_name = None
        if is_short_day_flag:
            prophet_event_name = "Предпраздничный"
        elif is_official_holiday_flag:  # Включаем все официальные праздники независимо от того, выходной день или нет
            prophet_event_name = holiday_name_pyholidays
        # Обычные выходные (Сб, Вс) и перенесенные непраздничные выходные не получают здесь имени,
        # Prophet обрабатывает их через недельную сезонность.

        all_days_data.append({
            'date': pd.to_datetime(current_dt_obj),
            'day_code_isdayoff': day_type_code_api,  # 0:раб, 1:вых/празд, 2:сокр, -1:нет данных
            'day_type_isdayoff': day_type_desc,  # 'workday', 'day_off', 'short_day', 'unknown'
            'official_holiday_name': holiday_name_pyholidays,  # Название из pyholidays (может быть None)
            'is_official_holiday': is_official_holiday_flag,  # Флаг официального праздника из pyholidays
            'is_day_off': is_day_off_flag,  # Выходной/праздник по данным API (или предположению)
            'is_short_day': is_short_day_flag,  # Сокращенный по данным API
            'prophet_holiday_event': prophet_event_name  # Имя для колонки 'holiday' в Prophet (может быть None)
        })

    return pd.DataFrame(all_days_data)


# --- Основной блок выполнения ---
if __name__ == "__main__":
    print(f"🚀 Запуск скрипта для формирования календаря праздников и выходных для РФ.")
    print(f"Диапазон дат: с 01.01.{START_YEAR} по 31.12.{END_YEAR}.")
    print("Источник данных о типах дней: API isdayoff.ru")
    print("Источник названий праздников: библиотека 'holidays' (pyholidays)")
    print("-" * 30)

    full_calendar_df = create_full_calendar_dataframe(START_YEAR, END_YEAR)

    if not full_calendar_df.empty:
        print("\n✅ Календарь успешно сформирован!")

        # Примеры записей из полного календаря
        print("\n📋 Примеры записей из полного календаря (calendar_df):")
        examples_df = pd.DataFrame()
        # Новый год
        new_year_ex = full_calendar_df[full_calendar_df['official_holiday_name'] == 'Новый год'].head(1)
        # Предпраздничный
        pre_holiday_ex = full_calendar_df[full_calendar_df['is_short_day'] == True].head(1)
        # Обычный выходной (не праздник, Сб/Вс)
        weekend_ex = full_calendar_df[
            (full_calendar_df['is_day_off'] == True) &
            (full_calendar_df['is_official_holiday'] == False) &
            (full_calendar_df['is_short_day'] == False) &
            (full_calendar_df['date'].dt.dayofweek.isin([5, 6]))  # 5=Сб, 6=Вс
            ].head(1)
        # Обычный рабочий день
        workday_ex = full_calendar_df[
            (full_calendar_df['is_day_off'] == False) &
            (full_calendar_df['is_short_day'] == False) &
            (full_calendar_df['is_official_holiday'] == False)
            ].head(1)

        example_list = [ex for ex in [new_year_ex, pre_holiday_ex, weekend_ex, workday_ex] if not ex.empty]
        if example_list:
            examples_df = pd.concat(example_list).drop_duplicates().reset_index(drop=True)
            print(examples_df)
        else:
            print("   Не удалось найти разнообразные примеры для отображения.")

        # Подготовка данных для Prophet
        # Проверяем, можем ли мы использовать готовый файл с праздниками
        try:
            prophet_holidays_df = pd.read_csv('prophet_holidays_2020_2027.csv')
            prophet_holidays_df['ds'] = pd.to_datetime(prophet_holidays_df['ds'])
            print(f"✅ Успешно загружен готовый файл prophet_holidays_2020_2027.csv с {len(prophet_holidays_df)} праздниками.")
        except Exception as e:
            print(f"ℹ️ Готовый файл с праздниками не найден или ошибка: {e}. Создаем новый файл.")
            # Если готовый файл недоступен, формируем из полного календаря
            prophet_holidays_df = full_calendar_df[full_calendar_df['prophet_holiday_event'].notna()].copy()
            prophet_holidays_df = prophet_holidays_df[['date', 'prophet_holiday_event']]
            prophet_holidays_df.rename(columns={'date': 'ds', 'prophet_holiday_event': 'holiday'}, inplace=True)

        # Проверяем, сколько праздников и предпраздничных дней попало в DataFrame
        predprazdnichniy_count = prophet_holidays_df[prophet_holidays_df['holiday'] == 'Предпраздничный'].shape[0]
        official_holidays_count = prophet_holidays_df[prophet_holidays_df['holiday'] != 'Предпраздничный'].shape[0]
        
        print(f"\n📊 Статистика по событиям для Prophet:")
        print(f"   - Официальных праздников: {official_holidays_count}")
        print(f"   - Предпраздничных дней: {predprazdnichniy_count}")
        print(f"   - Всего событий: {len(prophet_holidays_df)}")
        
        print("\n📊 Пример DataFrame для Prophet (только события 'holiday'):")
        if not prophet_holidays_df.empty:
            # Показываем примеры как предпраздничных дней, так и официальных праздников
            predprazdnichniy_sample = prophet_holidays_df[prophet_holidays_df['holiday'] == 'Предпраздничный'].head(2)
            holidays_sample = prophet_holidays_df[prophet_holidays_df['holiday'] != 'Предпраздничный'].head(3)
            
            sample_df = pd.concat([predprazdnichniy_sample, holidays_sample]).sort_values(by='ds')
            print(sample_df)
            
            # Проверяем наличие праздников в 2026 году
            holidays_2026 = prophet_holidays_df[prophet_holidays_df['ds'].dt.year == 2026]
            if not holidays_2026.empty:
                print(f"\n📊 Примеры событий для 2026 года (всего: {len(holidays_2026)}):")
                print(holidays_2026.head(3))
            else:
                print("\n⚠️ События для 2026 года отсутствуют!")
        else:
            print("   Событий для Prophet (праздники, предпраздничные) не найдено.")

        # Сохранение в CSV
        output_filename_full = f"russian_calendar_{START_YEAR}_{END_YEAR}.csv"
        full_calendar_df.to_csv(output_filename_full, index=False,
                                encoding='utf-8-sig')  # utf-8-sig для корректного Excel
        print(f"\n💾 Полный календарь сохранен в файл: {output_filename_full}")

        if not prophet_holidays_df.empty:
            # Всегда сохраняем с именем prophet_holidays_2020_2027.csv для консистентности
            output_filename_prophet = "prophet_holidays_2020_2027.csv"
            prophet_holidays_df.to_csv(output_filename_prophet, index=False, encoding='utf-8-sig')
            print(f"💾 Календарь для Prophet сохранен в файл: {output_filename_prophet}")
            
            # Также создаем копию с динамическим именем (если нужен именно такой формат)
            dynamic_filename = f"prophet_holidays_{START_YEAR}_{END_YEAR}.csv"
            if dynamic_filename != output_filename_prophet:
                prophet_holidays_df.to_csv(dynamic_filename, index=False, encoding='utf-8-sig')
                print(f"💾 Копия календаря для Prophet сохранена в файл: {dynamic_filename}")
        else:
            print(f"ℹ️ Файл для Prophet не создан, так как не найдено событий.")

        print("\n📌 Напоминание: Данный календарь является общероссийским. "
              "Если для Санкт-Петербурга существуют специфические нерабочие дни, не учтенные здесь, "
              "или важные городские события (не выходные), влияющие на трафик, их следует добавить отдельно.")

    else:
        print("\n❌ Не удалось создать DataFrame с календарем. Проверьте сообщения об ошибках выше.")

    print("\n🏁 Работа скрипта завершена.")