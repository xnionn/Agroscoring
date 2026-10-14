# =============================================================================
# ФАЗА 1: Загрузка данных и разведочный анализ (EDA)
# Система скоринга заявок на сельскохозяйственные субсидии — МСХ Казахстана
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# ШАГ 1: Загрузка Excel-файла
# header=4 → заголовки начинаются с 5-й строки (индекс 4)
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 1: Загрузка файла...")
print("=" * 60)

FILE_PATH = "Выгрузка_по_выданным_субсидиям_2025_год_обезлич.xlsx"

df_raw = pd.read_excel(FILE_PATH, header=4)
print(f"Загружено строк: {df_raw.shape[0]}, колонок: {df_raw.shape[1]}")
print(f"Исходные названия колонок:\n{list(df_raw.columns)}\n")

# Назначаем правильные имена колонок согласно документации датасета
# col0=num, col1=date, col4=region, col5=akim, col6=app_num,
# col7=direction, col8=subsidy_name, col9=status, col10=norm,
# col11=amount, col12=district
COLUMN_MAP = {
    df_raw.columns[0]:  "num",           # порядковый номер
    df_raw.columns[1]:  "date",          # дата заявки
    df_raw.columns[4]:  "region",        # регион
    df_raw.columns[5]:  "akim",          # аким (подписант)
    df_raw.columns[6]:  "app_num",       # номер заявки
    df_raw.columns[7]:  "direction",     # направление животноводства
    df_raw.columns[8]:  "subsidy_name",  # наименование субсидии
    df_raw.columns[9]:  "status",        # статус заявки
    df_raw.columns[10]: "norm",          # норматив (тг/голову)
    df_raw.columns[11]: "amount",        # сумма субсидии (тг)
    df_raw.columns[12]: "district",      # район
}
df_raw = df_raw.rename(columns=COLUMN_MAP)

# Оставляем только нужные колонки
KEEP_COLS = ["num", "date", "region", "akim", "app_num",
             "direction", "subsidy_name", "status", "norm", "amount", "district"]
df = df_raw[KEEP_COLS].copy()
print(f"Оставлено колонок: {len(KEEP_COLS)}")
print(f"Первые 3 строки после переименования:\n{df.head(3)}\n")

# ----------------------------------------------------------------------------
# ШАГ 2: Чистка данных
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 2: Чистка данных...")
print("=" * 60)

rows_before = len(df)

# 2.1 Убираем строки где 'num' не является числом (служебные/заголовочные строки)
df["num"] = pd.to_numeric(df["num"], errors="coerce")
df = df.dropna(subset=["num"])
df["num"] = df["num"].astype(int)
print(f"После удаления нечисловых 'num': {len(df)} строк (удалено {rows_before - len(df)})")

# 2.2 Конвертируем дату: dayfirst=True (формат ДД.ММ.ГГГГ)
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
invalid_dates = df["date"].isna().sum()
print(f"Некорректных дат: {invalid_dates}")

# 2.3 Конвертируем числовые поля amount и norm в float
# Убираем пробелы/запятые которые могут быть разделителями тысяч
for col in ["amount", "norm"]:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.replace(" ", "", regex=False)
        df[col] = df[col].str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors="coerce")

invalid_amount = df["amount"].isna().sum()
invalid_norm   = df["norm"].isna().sum()
print(f"Некорректных значений amount: {invalid_amount}")
print(f"Некорректных значений norm:   {invalid_norm}")

# 2.4 Убираем строки с нулевым или отрицательным нормативом (деление на ноль)
df = df[df["norm"].notna() & (df["norm"] > 0)]
print(f"Итого строк после чистки: {len(df)}\n")

# ----------------------------------------------------------------------------
# ШАГ 3: Базовая информация о датафрейме
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 3: Структура и качество данных")
print("=" * 60)

print(f"\nРазмер датафрейма: {df.shape[0]} строк × {df.shape[1]} колонок\n")

print("Типы данных:")
print(df.dtypes.to_string())

print("\nПроцент пропусков по каждой колонке:")
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_df = pd.DataFrame({"пропусков_шт": df.isnull().sum(), "пропусков_%": null_pct})
print(null_df.to_string())
print()

# ----------------------------------------------------------------------------
# ШАГ 4: value_counts для ключевых категориальных полей
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 4: Распределение категориальных переменных")
print("=" * 60)

print("\n--- Статусы заявок (status) ---")
status_vc = df["status"].value_counts()
print(status_vc.to_string())

print("\n--- Направления животноводства (direction) ---")
direction_vc = df["direction"].value_counts()
print(direction_vc.to_string())

print("\n--- Топ-10 регионов (region) ---")
region_vc = df["region"].value_counts().head(10)
print(region_vc.to_string())
print()

# ----------------------------------------------------------------------------
# ШАГ 5: Описательная статистика для amount и norm (в тысячах тенге)
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 5: Описательная статистика (тыс. тенге)")
print("=" * 60)

# Форматирование: делим на 1000 для наглядности
def fmt_thousands(series):
    """Возвращает describe() в тысячах тенге с форматированием."""
    desc = (series / 1_000).describe()
    return desc.apply(lambda x: f"{x:,.2f} тыс. тг")

print("\n--- Сумма субсидии (amount) ---")
print(fmt_thousands(df["amount"]).to_string())

print("\n--- Норматив на голову (norm) ---")
print(fmt_thousands(df["norm"]).to_string())
print()

# ----------------------------------------------------------------------------
# ШАГ 6: Расчёт поголовья скота (косвенный признак)
# heads = amount / norm — приблизительное количество голов скота
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 6: Поголовье скота (heads = amount / norm)")
print("=" * 60)

df["heads"] = df["amount"] / df["norm"]

print("\n--- Описательная статистика: heads (голов) ---")
heads_desc = df["heads"].describe()
print(heads_desc.apply(lambda x: f"{x:,.1f}").to_string())

# Выводим медиану и 95-й перцентиль для понимания распределения
print(f"\nМедиана поголовья:          {df['heads'].median():,.1f} голов")
print(f"95-й перцентиль поголовья:  {df['heads'].quantile(0.95):,.1f} голов")
print(f"Максимум поголовья:         {df['heads'].max():,.1f} голов\n")

# ----------------------------------------------------------------------------
# ШАГ 7: Построение 4 Plotly-графиков
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 7: Построение графиков...")
print("=" * 60)

# --- 7.1 Bar chart: количество заявок по статусам ---
print("  → График 1: Статусы заявок")

# Словарь цветов: зелёный=Исполнена, красный=Отклонена, серый=остальные
STATUS_COLORS = {
    "Исполнена":                  "#2ECC71",
    "Отклонена":                  "#E74C3C",
    "Одобрена":                   "#95A5A6",
    "Отозвано":                   "#BDC3C7",
    "Сформировано поручение":     "#AAB7B8",
    "Получена":                   "#D5D8DC",
}

status_counts = df["status"].value_counts().reset_index()
status_counts.columns = ["status", "count"]
status_counts["color"] = status_counts["status"].map(
    lambda s: STATUS_COLORS.get(s, "#BDC3C7")
)

fig1 = go.Figure(go.Bar(
    x=status_counts["status"],
    y=status_counts["count"],
    marker_color=status_counts["color"],
    text=status_counts["count"].apply(lambda x: f"{x:,}"),
    textposition="outside",
))
fig1.update_layout(
    title="Распределение заявок по статусам",
    xaxis_title="Статус заявки",
    yaxis_title="Количество заявок",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=13),
    showlegend=False,
    margin=dict(t=60, b=120),
)
fig1.update_xaxes(tickangle=-20)
fig1.write_html("chart1_statuses.html")
print("     Сохранено: chart1_statuses.html")

# --- 7.2 Bar chart: топ-10 регионов по количеству заявок ---
print("  → График 2: Топ-10 регионов")

top10_regions = df["region"].value_counts().head(10).reset_index()
top10_regions.columns = ["region", "count"]
top10_regions = top10_regions.sort_values("count", ascending=True)  # для горизонтального бара

fig2 = go.Figure(go.Bar(
    x=top10_regions["count"],
    y=top10_regions["region"],
    orientation="h",
    marker_color="#3498DB",
    text=top10_regions["count"].apply(lambda x: f"{x:,}"),
    textposition="outside",
))
fig2.update_layout(
    title="Топ-10 регионов по количеству заявок",
    xaxis_title="Количество заявок",
    yaxis_title="Регион",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=13),
    showlegend=False,
    margin=dict(t=60, l=200),
    height=450,
)
fig2.write_html("chart2_regions.html")
print("     Сохранено: chart2_regions.html")

# --- 7.3 Histogram: распределение суммы субсидий (без топ 1% выбросов) ---
print("  → График 3: Распределение суммы субсидий")

# Срезаем верхний 1% выбросов для читаемого графика
p99 = df["amount"].quantile(0.99)
df_hist = df[df["amount"] <= p99].copy()
df_hist["amount_mln"] = df_hist["amount"] / 1_000_000  # переводим в млн тенге

fig3 = px.histogram(
    df_hist,
    x="amount_mln",
    nbins=60,
    color_discrete_sequence=["#9B59B6"],
    labels={"amount_mln": "Сумма субсидии (млн тенге)"},
    title="Распределение суммы субсидий (без топ 1% выбросов)",
)
fig3.update_layout(
    yaxis_title="Количество заявок",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=13),
    bargap=0.05,
)
fig3.add_annotation(
    text=f"Показано {len(df_hist):,} из {len(df):,} заявок (99-й перцентиль: {p99/1e6:.1f} млн тг)",
    xref="paper", yref="paper", x=0.98, y=0.95,
    showarrow=False, font=dict(size=10, color="grey"), align="right",
)
fig3.write_html("chart3_amount_hist.html")
print("     Сохранено: chart3_amount_hist.html")

# --- 7.4 Bar chart: направления животноводства ---
print("  → График 4: Направления животноводства")

direction_counts = df["direction"].value_counts().reset_index()
direction_counts.columns = ["direction", "count"]
direction_counts["pct"] = (direction_counts["count"] / direction_counts["count"].sum() * 100).round(1)

DIRECTION_COLORS = [
    "#E67E22", "#F39C12", "#D4AC0D", "#A9CCE3",
    "#85C1E9", "#AED6F1", "#5DADE2", "#2E86C1",
]

fig4 = go.Figure(go.Bar(
    x=direction_counts["direction"],
    y=direction_counts["count"],
    marker_color=DIRECTION_COLORS[:len(direction_counts)],
    text=direction_counts.apply(lambda r: f"{r['count']:,}<br>({r['pct']}%)", axis=1),
    textposition="outside",
))
fig4.update_layout(
    title="Распределение заявок по направлениям животноводства",
    xaxis_title="Направление",
    yaxis_title="Количество заявок",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=13),
    showlegend=False,
    margin=dict(t=60, b=150),
)
fig4.update_xaxes(tickangle=-30)
fig4.write_html("chart4_directions.html")
print("     Сохранено: chart4_directions.html\n")

# ----------------------------------------------------------------------------
# ШАГ 8: Сохранение очищенного датафрейма
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 8: Сохранение очищенного датафрейма...")
print("=" * 60)

with open("cleaned_data.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Файл 'cleaned_data.pkl' сохранён.")
print(f"Итоговый датафрейм: {df.shape[0]:,} строк × {df.shape[1]} колонок")
print(f"Колонки: {list(df.columns)}")

# ----------------------------------------------------------------------------
# ФИНАЛЬНАЯ СВОДКА
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ФИНАЛЬНАЯ СВОДКА EDA")
print("=" * 60)

total      = len(df)
ispolnena  = (df["status"] == "Исполнена").sum()
otklonena  = (df["status"] == "Отклонена").sum()
total_mln  = df["amount"].sum() / 1_000_000_000  # в млрд тенге

print(f"Всего заявок после чистки:      {total:,}")
print(f"  → Исполнена (y=1):            {ispolnena:,}  ({ispolnena/total*100:.1f}%)")
print(f"  → Отклонена (y=0):            {otklonena:,}  ({otklonena/total*100:.1f}%)")
print(f"  → Обучающая выборка (1+0):    {ispolnena+otklonena:,}")
print(f"Итого выдано субсидий:          {total_mln:.1f} млрд тенге")
print(f"Уникальных регионов:            {df['region'].nunique()}")
print(f"Уникальных направлений:         {df['direction'].nunique()}")
print(f"Период данных:                  {df['date'].min().date()} — {df['date'].max().date()}")
print("\nФаза 1 завершена успешно!")
