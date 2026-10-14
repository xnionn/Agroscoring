# =============================================================================
# ФАЗА 2: Feature Engineering
# Система скоринга заявок на сельскохозяйственные субсидии — МСХ Казахстана
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# ШАГ 1: Загрузка очищенного датафрейма из Фазы 1
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 1: Загрузка cleaned_data.pkl...")
print("=" * 60)

with open("cleaned_data.pkl", "rb") as f:
    df = pickle.load(f)

print(f"Загружено: {df.shape[0]:,} строк × {df.shape[1]} колонок")
print(f"Колонки: {list(df.columns)}\n")

# ----------------------------------------------------------------------------
# ШАГ 2: Формирование целевой переменной
# y = 1 → "Исполнена", y = 0 → "Отклонена"
# Остальные статусы исключаем из обучающей выборки, но сохраняем в df
# для расчёта бенчмарков по всей базе
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 2: Целевая переменная...")
print("=" * 60)

# Маска строк, используемых для расчёта региональных бенчмарков
# (только Исполнена + Отклонена — «завершённые» заявки)
COMPLETED_MASK = df["status"].isin(["Исполнена", "Отклонена"])
df_completed = df[COMPLETED_MASK].copy()

# Целевая переменная только для завершённых заявок
df_completed["target"] = (df_completed["status"] == "Исполнена").astype(int)

print(f"Обучающая выборка (Исполнена + Отклонена): {len(df_completed):,} строк")
print(f"  y=1 (Исполнена): {df_completed['target'].sum():,}")
print(f"  y=0 (Отклонена): {(df_completed['target'] == 0).sum():,}")
print(f"  Баланс классов:  {df_completed['target'].mean():.3f} (доля одобрений)\n")

# ----------------------------------------------------------------------------
# ШАГ 3: ЧИСЛОВЫЕ ПРИЗНАКИ
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 3: Числовые признаки...")
print("=" * 60)

# heads уже посчитан в Фазе 1, но пересчитаем для надёжности
# Поголовье скота: косвенный показатель масштаба хозяйства
df["heads"] = df["amount"] / df["norm"]

# Логарифмическое преобразование суммы субсидии
# Убирает правый хвост (сильную правостороннюю асимметрию)
df["log_amount"] = np.log1p(df["amount"])

# Логарифм поголовья — аналогично для heads
df["log_heads"] = np.log1p(df["heads"])

# Сумма заявителя относительно среднего по его региону
# >1 означает "запрашивает больше среднего по региону"
region_mean_amount = df.groupby("region")["amount"].transform("mean")
df["amount_vs_region_avg"] = df["amount"] / region_mean_amount

# Поголовье относительно среднего по направлению
# Позволяет сравнивать скотоводов между собой, птицеводов между собой и т.д.
direction_mean_heads = df.groupby("direction")["heads"].transform("mean")
df["heads_vs_direction_avg"] = df["heads"] / direction_mean_heads

# Норматив относительно среднего по направлению
# Высокий норматив может указывать на субсидию с повышенными требованиями
direction_mean_norm = df.groupby("direction")["norm"].transform("mean")
df["norm_vs_direction_avg"] = df["norm"] / direction_mean_norm

print("  ✓ heads, log_amount, log_heads")
print("  ✓ amount_vs_region_avg")
print("  ✓ heads_vs_direction_avg")
print("  ✓ norm_vs_direction_avg\n")

# ----------------------------------------------------------------------------
# ШАГ 4: РЕГИОНАЛЬНЫЕ БЕНЧМАРКИ
# Считаем ТОЛЬКО на завершённых заявках (Исполнена + Отклонена),
# чтобы не допустить data leakage из незавершённых статусов
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 4: Региональные бенчмарки...")
print("=" * 60)

# --- 4.1 region_approval_rate: % одобрения по региону ---
# Показывает насколько лояльна комиссия в данном регионе
region_rate = (
    df_completed.groupby("region")["target"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "region_approval_rate", "count": "region_application_count"})
    .reset_index()
)

# --- 4.2 direction_approval_rate: % одобрения по направлению ---
# Некоторые направления могут иметь системно более высокий % одобрения
direction_rate = (
    df_completed.groupby("direction")["target"]
    .mean()
    .rename("direction_approval_rate")
    .reset_index()
)

# --- 4.3 district_approval_rate: % одобрения по району ---
# Более локальный бенчмарк — внутри региона могут быть «лояльные» районы
district_rate = (
    df_completed.groupby("district")["target"]
    .mean()
    .rename("district_approval_rate")
    .reset_index()
)

# --- 4.4 region_avg_amount: средняя сумма субсидии по региону ---
# (считаем по всему датафрейму — это не утечка, нет связи с target)
region_avg_amount = (
    df.groupby("region")["amount"]
    .mean()
    .rename("region_avg_amount")
    .reset_index()
)

# Джойн всех бенчмарков к исходному датафрейму
df = df.merge(region_rate, on="region", how="left")
df = df.merge(direction_rate, on="direction", how="left")
df = df.merge(district_rate, on="district", how="left")
df = df.merge(region_avg_amount, on="region", how="left")

# Заполняем пропуски (районы без истории) медианой
df["district_approval_rate"] = df["district_approval_rate"].fillna(
    df["district_approval_rate"].median()
)

print("  ✓ region_approval_rate")
print("  ✓ direction_approval_rate")
print("  ✓ district_approval_rate (NaN →  медиана)")
print("  ✓ region_avg_amount")
print("  ✓ region_application_count\n")

# ----------------------------------------------------------------------------
# ШАГ 5: ВРЕМЕННЫЕ ПРИЗНАКИ
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 5: Временные признаки...")
print("=" * 60)

# Месяц подачи: зимние/весенние заявки могут иметь разный характер
df["month"] = df["date"].dt.month

# День недели (0=Пн, 6=Вс): поданные в будни могут отличаться от выходных
df["day_of_week"] = df["date"].dt.dayofweek

# Час подачи: ранние утренние/поздние вечерние заявки могут быть аномалиями
df["hour"] = df["date"].dt.hour

# Признак "начало месяца": первые 10 дней — пик активности подач
df["is_early_month"] = (df["date"].dt.day <= 10).astype(int)

print("  ✓ month, day_of_week, hour")
print("  ✓ is_early_month (1 если день ≤ 10)\n")

# ----------------------------------------------------------------------------
# ШАГ 6: КАТЕГОРИАЛЬНЫЙ ПРИЗНАК — subsidy_category
# Группируем типы субсидий в смысловые категории по ключевым словам
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 6: Категория субсидии (subsidy_category)...")
print("=" * 60)

def classify_subsidy(name: str) -> str:
    """
    Классифицирует субсидию по ключевым словам в наименовании.
    Приоритет: сверху вниз — первое совпадение побеждает.
    ВАЖНО: "приобретение" проверяем ДО "племенная_работа",
    т.к. "приобретение племенных быков" содержит оба keyword.
    """
    if pd.isna(name):
        return "прочее"
    name_lower = str(name).lower()
    # Осеменение: семя / искусственное осеменение
    if "осеменени" in name_lower or "семя" in name_lower:
        return "осеменение"
    # Приобретение: покупка скота / оборудования
    if "приобретени" in name_lower or "приобретен" in name_lower:
        return "приобретение"
    # Удешевление: компенсация себестоимости
    if "удешевлени" in name_lower:
        return "удешевление"
    # Племенная работа: селекция, ведение племенной работы
    if "племенной работ" in name_lower or "селекционной и племенной" in name_lower:
        return "племенная_работа"
    return "прочее"

df["subsidy_category"] = df["subsidy_name"].apply(classify_subsidy)

print("  ✓ subsidy_category — распределение:")
print(df["subsidy_category"].value_counts().to_string())
print()

# ----------------------------------------------------------------------------
# ШАГ 7: КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
# LabelEncoder: direction, region, district, subsidy_category
# Сохраняем encoders для последующего использования в инференсе
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 7: Label Encoding...")
print("=" * 60)

CAT_COLS   = ["direction", "region", "district", "subsidy_category"]
encoders   = {}  # словарь {колонка: fitted LabelEncoder}

for col in CAT_COLS:
    le = LabelEncoder()
    # Заполняем возможные NaN строкой "Unknown" перед кодированием
    df[col + "_enc"] = le.fit_transform(df[col].fillna("Unknown").astype(str))
    encoders[col] = le
    print(f"  ✓ {col}_enc  ({le.classes_.shape[0]} уникальных значений)")

# Сохраняем словарь энкодеров для Фазы 3 (инференс / Streamlit)
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
print("\n  Encoders сохранены: encoders.pkl\n")

# ----------------------------------------------------------------------------
# ШАГ 8: Финальный список признаков и сохранение датафрейма
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 8: Сохранение features_data.pkl...")
print("=" * 60)

# Полный список фич для модели
FEATURE_COLS = [
    # Числовые
    "log_amount",
    "log_heads",
    "amount_vs_region_avg",
    "heads_vs_direction_avg",
    "norm_vs_direction_avg",
    # Региональные бенчмарки
    "region_approval_rate",
    "direction_approval_rate",
    "district_approval_rate",
    "region_avg_amount",
    "region_application_count",
    # Временные
    "month",
    "day_of_week",
    "hour",
    "is_early_month",
    # Закодированные категории
    "direction_enc",
    "region_enc",
    "district_enc",
    "subsidy_category_enc",
]

print(f"\nИтоговый список фич ({len(FEATURE_COLS)} штук):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {f}")

# Добавляем target в датафрейм (NaN для незавершённых статусов)
df["target"] = np.nan
df.loc[df["status"] == "Исполнена", "target"] = 1
df.loc[df["status"] == "Отклонена", "target"] = 0

# Проверяем на Inf (деление на ноль в relative-признаках)
for col in FEATURE_COLS:
    n_inf = np.isinf(df[col]).sum()
    if n_inf > 0:
        median_val = df.loc[~np.isinf(df[col]), col].median()
        df[col] = df[col].replace([np.inf, -np.inf], median_val)
        print(f"  ! Заменено {n_inf} Inf в '{col}' на медиану ({median_val:.4f})")

# Сохраняем полный датафрейм со всеми фичами
with open("features_data.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nСохранено: features_data.pkl  ({df.shape[0]:,} строк × {df.shape[1]} колонок)")

# ----------------------------------------------------------------------------
# ШАГ 9: Корреляционная матрица топ-15 фич с target (Plotly Heatmap)
# Считаем только на завершённых заявках (target не NaN)
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 9: Корреляционная матрица топ-15 фич...")
print("=" * 60)

# Берём только строки с известным target
df_train = df[df["target"].notna()].copy()

# Считаем корреляцию Пирсона каждой фичи с target
corr_with_target = (
    df_train[FEATURE_COLS + ["target"]]
    .corr()["target"]
    .drop("target")
    .abs()
    .sort_values(ascending=False)
)

# Топ-15 фич по абсолютной корреляции с target
TOP15 = corr_with_target.head(15).index.tolist()
print(f"\nТоп-15 фич по корреляции с target:")
for i, feat in enumerate(TOP15, 1):
    raw_corr = df_train[FEATURE_COLS + ["target"]].corr()["target"][feat]
    print(f"  {i:2d}. {feat:<35s}  r = {raw_corr:+.4f}")

# Строим полную корреляционную матрицу между топ-15 фичами + target
corr_matrix = df_train[TOP15 + ["target"]].corr()

# Переименовываем "target" для красоты оси
labels = TOP15 + ["TARGET"]
corr_values = corr_matrix.values

fig = go.Figure(go.Heatmap(
    z=corr_values,
    x=labels,
    y=labels,
    colorscale="RdBu",
    zmid=0,
    zmin=-1,
    zmax=1,
    text=np.round(corr_values, 2),
    texttemplate="%{text}",
    textfont={"size": 9},
    colorbar=dict(title="Корреляция Пирсона"),
))

fig.update_layout(
    title="Корреляционная матрица топ-15 признаков с целевой переменной",
    xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    width=900,
    height=850,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=11),
    margin=dict(t=70, l=180, b=180),
)

fig.write_html("chart5_correlation_heatmap.html")
print("\nСохранено: chart5_correlation_heatmap.html")

# ----------------------------------------------------------------------------
# ФИНАЛЬНАЯ СВОДКА ФАЗЫ 2
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ФИНАЛЬНАЯ СВОДКА ФАЗЫ 2")
print("=" * 60)
print(f"Всего признаков создано:         {len(FEATURE_COLS)}")
print(f"Числовых:                        5")
print(f"Региональных бенчмарков:         5")
print(f"Временных:                       4")
print(f"Закодированных категориальных:   4")
print(f"Обучающая выборка (target≠NaN):  {len(df_train):,} строк")
print(f"Файлы: features_data.pkl, encoders.pkl, chart5_correlation_heatmap.html")
print("\nФаза 2 завершена успешно! Готов к Фазе 3: XGBoost + SHAP")
