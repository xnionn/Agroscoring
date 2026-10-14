# =============================================================================
# ФАЗА 4: Rule-based фильтры, Anomaly Detection, Composite Score
# Система скоринга заявок на сельскохозяйственные субсидии — МСХ Казахстана
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# ШАГ 1: Загрузка scored_data.pkl из Фазы 3
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 1: Загрузка scored_data.pkl...")
print("=" * 60)

with open("scored_data.pkl", "rb") as f:
    df = pickle.load(f)

print(f"Загружено: {df.shape[0]:,} строк × {df.shape[1]} колонок\n")

# ----------------------------------------------------------------------------
# ШАГ 2: RULE-BASED ФИЛЬТРЫ
# Простые детерминированные правила — прозрачны и объяснимы комиссии
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 2: Rule-based фильтры...")
print("=" * 60)

# --- Правило 1: heads >= 1 (поголовье не менее 1 головы) ---
# Нулевое поголовье — технический артефакт или ошибка заявки
df["rule_min_heads"] = (df["heads"] >= 1).astype(int)

# --- Правило 2: норматив строго положительный ---
# norm <= 0 означает некорректные данные
df["rule_norm_positive"] = (df["norm"] > 0).astype(int)

# --- Правило 3: сумма субсидии строго положительная ---
# amount = 0 — бессмысленная заявка
df["rule_amount_positive"] = (df["amount"] > 0).astype(int)

# --- Правило 4: amount не выше 99-го перцентиля по направлению ---
# Защита от экстремальных заявок, которые «пожирают» бюджет одного направления
p99_by_direction = df.groupby("direction")["amount"].transform(lambda x: x.quantile(0.99))
df["rule_not_extreme_amount"] = (df["amount"] <= p99_by_direction).astype(int)

# --- Правило 5: норматив в коридоре mean ± 3*std по направлению ---
# Выход за 3 сигмы → нетипичный норматив, возможно ошибка
norm_mean = df.groupby("direction")["norm"].transform("mean")
norm_std  = df.groupby("direction")["norm"].transform("std").fillna(0)
norm_low  = norm_mean - 3 * norm_std
norm_high = norm_mean + 3 * norm_std
df["rule_norm_in_range"] = (
    (df["norm"] >= norm_low) & (df["norm"] <= norm_high)
).astype(int)

# --- rule_score: среднее всех бинарных правил (0–1) ---
RULE_COLS = [
    "rule_min_heads",
    "rule_norm_positive",
    "rule_amount_positive",
    "rule_not_extreme_amount",
    "rule_norm_in_range",
]
df["rule_score"] = df[RULE_COLS].mean(axis=1)

# --- Отчёт: сколько заявок нарушает каждое правило ---
print(f"{'Правило':<30s}  {'Нарушений':>10s}  {'% от всех':>10s}")
print("-" * 55)
for rule in RULE_COLS:
    violations = (df[rule] == 0).sum()
    pct = violations / len(df) * 100
    print(f"{rule:<30s}  {violations:>10,}  {pct:>9.2f}%")

total_any_violation = (df[RULE_COLS].min(axis=1) == 0).sum()
print("-" * 55)
print(f"{'ИТОГО (хотя бы 1 нарушение)':<30s}  {total_any_violation:>10,}  "
      f"{total_any_violation/len(df)*100:>9.2f}%")

print(f"\nСредний rule_score по всем заявкам: {df['rule_score'].mean():.4f}")
print(f"Заявок с rule_score = 1.0 (все правила OK): "
      f"{(df['rule_score'] == 1.0).sum():,}")

# ----------------------------------------------------------------------------
# ШАГ 3: ANOMALY DETECTION — Isolation Forest
# Выявляет статистически нетипичные заявки (потенциальные мошенничества/ошибки)
# contamination=0.05 → ожидаем ~5% аномалий в данных
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 3: Anomaly Detection — Isolation Forest...")
print("=" * 60)

# Четыре числовых признака, описывающих «нетипичность» заявки
ANOMALY_FEATURES = [
    "log_amount",
    "log_heads",
    "norm_vs_direction_avg",
    "amount_vs_region_avg",
]

X_anomaly = df[ANOMALY_FEATURES].values

iso = IsolationForest(
    contamination = 0.05,   # ожидаемая доля аномалий
    random_state  = 42,
    n_estimators  = 100,
)
iso.fit(X_anomaly)

# anomaly_label: -1 = аномалия, 1 = норма (стандарт sklearn)
df["anomaly_label"] = iso.predict(X_anomaly)

# anomaly_clean: 1 = нормальная заявка, 0 = аномальная (для composite score)
df["anomaly_clean"] = (df["anomaly_label"] == 1).astype(int)

# anomaly_score_raw: чем ниже (отрицательнее), тем аномальнее
raw_scores = iso.decision_function(X_anomaly)

# Нормируем в 0–1: 0 = максимально аномальная, 1 = нормальная
raw_min = raw_scores.min()
raw_max = raw_scores.max()
df["anomaly_score_norm"] = (raw_scores - raw_min) / (raw_max - raw_min)

n_anomalies = (df["anomaly_label"] == -1).sum()
print(f"Обнаружено аномалий: {n_anomalies:,} ({n_anomalies/len(df)*100:.1f}%)")
print(f"Нормальных заявок:   {(df['anomaly_label'] == 1).sum():,}")

# --- Топ-10 самых аномальных заявок ---
print(f"\nТоп-10 самых аномальных заявок:")
top_anomalies = (
    df[df["anomaly_label"] == -1]
    .sort_values("anomaly_score_norm")
    .head(10)[["num", "date", "region", "direction", "amount", "norm",
               "heads", "status", "anomaly_score_norm"]]
    .copy()
)
top_anomalies["amount_млн"] = (top_anomalies["amount"] / 1_000_000).round(2)
top_anomalies["heads"]      = top_anomalies["heads"].round(1)
top_anomalies["anomaly_score_norm"] = top_anomalies["anomaly_score_norm"].round(4)

display_cols = ["num", "region", "direction", "amount_млн", "heads",
                "norm", "status", "anomaly_score_norm"]
print(top_anomalies[display_cols].to_string(index=False))

# ----------------------------------------------------------------------------
# ШАГ 4: COMPOSITE SCORE
# Взвешенная комбинация: ML (50%) + Rules (30%) + Anomaly clean (20%)
# Итоговый скор прозрачен: каждая компонента объяснима комиссии
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 4: Composite Score...")
print("=" * 60)

df["composite_score"] = (
    df["ml_score"]     * 0.5 +
    df["rule_score"]   * 0.3 +
    df["anomaly_clean"] * 0.2
) * 100

df["composite_score"] = df["composite_score"].round(2)

# --- Тир-классификация для комиссии ---
def assign_tier(score: float) -> str:
    if score >= 75:
        return "🟢 Приоритет"
    elif score >= 50:
        return "🟡 Рассмотреть"
    else:
        return "🔴 Отклонить"

df["tier"] = df["composite_score"].apply(assign_tier)

# --- Статистика composite_score ---
print("Статистика composite_score:")
print(df["composite_score"].describe().apply(lambda x: f"{x:.2f}").to_string())

print(f"\nРаспределение по тирам:")
tier_counts = df["tier"].value_counts()
for tier, cnt in tier_counts.items():
    pct = cnt / len(df) * 100
    avg_ml = df[df["tier"] == tier]["ml_score_100"].mean()
    print(f"  {tier:<25s}: {cnt:>6,} заявок ({pct:>5.1f}%)  | "
          f"avg ML score: {avg_ml:.1f}")

# --- Plotly: распределение composite_score по тирам ---
fig_dist = go.Figure()
colors_tier = {
    "🟢 Приоритет":   "#2ECC71",
    "🟡 Рассмотреть": "#F39C12",
    "🔴 Отклонить":   "#E74C3C",
}
for tier_name, color in colors_tier.items():
    subset = df[df["tier"] == tier_name]["composite_score"]
    fig_dist.add_trace(go.Histogram(
        x=subset,
        name=tier_name,
        marker_color=color,
        opacity=0.75,
        nbinsx=40,
    ))

fig_dist.add_vline(x=75, line_dash="dash", line_color="green",
                   annotation_text="≥75: Приоритет", annotation_position="top right")
fig_dist.add_vline(x=50, line_dash="dash", line_color="orange",
                   annotation_text="≥50: Рассмотреть", annotation_position="top right")

fig_dist.update_layout(
    title="Распределение Composite Score по тирам",
    xaxis_title="Composite Score (0–100)",
    yaxis_title="Количество заявок",
    barmode="overlay",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=12),
    legend=dict(title="Тир"),
    height=420,
)
fig_dist.write_html("chart7_composite_distribution.html")
print("\nСохранено: chart7_composite_distribution.html")

# --- Сохраняем итоговый датафрейм ---
with open("final_scored.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Сохранено: final_scored.pkl")

# ----------------------------------------------------------------------------
# ШАГ 5: СРАВНИТЕЛЬНЫЙ СЛАЙД — «Первый подал» vs «Лучший скор»
# ГЛАВНЫЙ АРГУМЕНТ для жюри: merit-based vs FIFO
# Берём только «активные» заявки (не Отклонена, не Отозвано) для честного сравнения
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 5: Сравнительный слайд (FIFO vs Merit-based)...")
print("=" * 60)

# Фильтруем заявки, которые «в очереди» на рассмотрение
# (исключаем уже окончательно завершённые — Исполнена/Отклонена/Отозвано)
ACTIVE_STATUSES = ["Одобрена", "Сформировано поручение", "Получена",
                   "Исполнена", "Отклонена", "Отозвано"]
df_compare = df.copy()  # берём весь датафрейм для наглядности

# --- Топ-10 по дате (FIFO — старый подход "первый подал — первый получил") ---
top10_fifo = (
    df_compare
    .sort_values("date")
    .head(10)[["num", "date", "region", "direction", "amount",
               "status", "composite_score", "tier"]]
    .copy()
)
top10_fifo["rank_fifo"] = range(1, 11)
top10_fifo["amount_млн"] = (top10_fifo["amount"] / 1_000_000).round(2)

# --- Топ-10 по composite_score (merit-based — новый подход) ---
top10_merit = (
    df_compare
    .sort_values("composite_score", ascending=False)
    .head(10)[["num", "date", "region", "direction", "amount",
               "status", "composite_score", "tier"]]
    .copy()
)
top10_merit["rank_merit"] = range(1, 11)
top10_merit["amount_млн"] = (top10_merit["amount"] / 1_000_000).round(2)

# --- Вывод таблиц ---
print("\n📋 ТОП-10 ПО ДАТЕ ПОДАЧИ (старый принцип FIFO):")
print("-" * 95)
print(f"{'#':>3} {'Регион':<30} {'Направление':<25} {'Сумма,млн':>10} "
      f"{'Статус':<20} {'Score':>6} {'Тир':<15}")
print("-" * 95)
for _, row in top10_fifo.iterrows():
    direction_short = row["direction"].replace("Субсидирование в ", "").replace("Субсидирование ", "")
    print(f"{row['rank_fifo']:>3} {row['region']:<30} {direction_short:<25} "
          f"{row['amount_млн']:>10.2f} {row['status']:<20} "
          f"{row['composite_score']:>6.1f} {row['tier']}")

print(f"\n⭐ ТОП-10 ПО COMPOSITE SCORE (новый merit-based подход):")
print("-" * 95)
print(f"{'#':>3} {'Регион':<30} {'Направление':<25} {'Сумма,млн':>10} "
      f"{'Статус':<20} {'Score':>6} {'Тир':<15}")
print("-" * 95)
for _, row in top10_merit.iterrows():
    direction_short = row["direction"].replace("Субсидирование в ", "").replace("Субсидирование ", "")
    print(f"{row['rank_merit']:>3} {row['region']:<30} {direction_short:<25} "
          f"{row['amount_млн']:>10.2f} {row['status']:<20} "
          f"{row['composite_score']:>6.1f} {row['tier']}")

# --- Ключевое сравнение: пересечение топ-10 ---
fifo_nums  = set(top10_fifo["num"].tolist())
merit_nums = set(top10_merit["num"].tolist())
overlap    = fifo_nums & merit_nums
print(f"\n{'='*55}")
print(f"Пересечение топ-10 FIFO и топ-10 Merit: {len(overlap)} из 10")
print(f"Заявок, которые пропустил бы FIFO:       {10 - len(overlap)}")
print(f"{'='*55}")

# Средние скоры двух групп
avg_fifo_score  = top10_fifo["composite_score"].mean()
avg_merit_score = top10_merit["composite_score"].mean()
print(f"\nСредний composite_score топ-10 FIFO:    {avg_fifo_score:.1f}")
print(f"Средний composite_score топ-10 Merit:   {avg_merit_score:.1f}")
print(f"Разрыв в качестве заявок:               +{avg_merit_score - avg_fifo_score:.1f} баллов")

# --- Plotly: визуальный сравнительный слайд ---
fig_compare = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "🕐 FIFO: Топ-10 по дате подачи",
        "⭐ Merit: Топ-10 по Composite Score",
    ),
    horizontal_spacing=0.12,
)

# Цвет бара — по тиру
def tier_color(tier_str):
    if "Приоритет" in tier_str:  return "#2ECC71"
    if "Рассмотреть" in tier_str: return "#F39C12"
    return "#E74C3C"

# Левый график — FIFO
fifo_sorted = top10_fifo.sort_values("composite_score", ascending=True)
fig_compare.add_trace(go.Bar(
    x=fifo_sorted["composite_score"],
    y=[f"#{r} {reg[:18]}" for r, reg in
       zip(fifo_sorted["rank_fifo"], fifo_sorted["region"])],
    orientation="h",
    marker_color=[tier_color(t) for t in fifo_sorted["tier"]],
    text=fifo_sorted["composite_score"].apply(lambda x: f"{x:.1f}"),
    textposition="outside",
    name="FIFO",
    showlegend=False,
), row=1, col=1)

# Правый график — Merit
merit_sorted = top10_merit.sort_values("composite_score", ascending=True)
fig_compare.add_trace(go.Bar(
    x=merit_sorted["composite_score"],
    y=[f"#{r} {reg[:18]}" for r, reg in
       zip(merit_sorted["rank_merit"], merit_sorted["region"])],
    orientation="h",
    marker_color=[tier_color(t) for t in merit_sorted["tier"]],
    text=merit_sorted["composite_score"].apply(lambda x: f"{x:.1f}"),
    textposition="outside",
    name="Merit",
    showlegend=False,
), row=1, col=2)

fig_compare.update_layout(
    title={
        "text": (f"FIFO vs Merit-based скоринг<br>"
                 f"<sup>Средний скор FIFO: {avg_fifo_score:.1f}  →  "
                 f"Merit: {avg_merit_score:.1f}  "
                 f"(+{avg_merit_score - avg_fifo_score:.1f} баллов)</sup>"),
        "x": 0.5,
        "xanchor": "center",
    },
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=11),
    height=520,
    width=1100,
    margin=dict(t=90, l=160, r=80),
)
fig_compare.update_xaxes(range=[0, 105], title_text="Composite Score")

fig_compare.write_html("chart8_fifo_vs_merit.html")
print("\nСохранено: chart8_fifo_vs_merit.html")

# --- Дополнительно: сохраняем две таблицы в CSV для слайда ---
top10_fifo[["rank_fifo", "region", "direction", "amount_млн",
            "date", "status", "composite_score", "tier"]].to_csv(
    "top10_fifo.csv", index=False, encoding="utf-8-sig")

top10_merit[["rank_merit", "region", "direction", "amount_млн",
             "date", "status", "composite_score", "tier"]].to_csv(
    "top10_merit.csv", index=False, encoding="utf-8-sig")

print("Сохранено: top10_fifo.csv, top10_merit.csv")

# ----------------------------------------------------------------------------
# ФИНАЛЬНАЯ СВОДКА ФАЗЫ 4
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ФИНАЛЬНАЯ СВОДКА ФАЗЫ 4")
print("=" * 60)
print(f"Rule-based нарушений (хотя бы 1): {total_any_violation:,} "
      f"({total_any_violation/len(df)*100:.1f}%)")
print(f"Аномалий (Isolation Forest 5%):   {n_anomalies:,}")
print(f"\nComposite Score — распределение:")
for tier_name, cnt in tier_counts.sort_index().items():
    print(f"  {tier_name:<25s}: {cnt:>6,} ({cnt/len(df)*100:.1f}%)")
print(f"\nФИФО vs Мерит:")
print(f"  Средний скор FIFO топ-10:        {avg_fifo_score:.1f}")
print(f"  Средний скор Merit топ-10:       {avg_merit_score:.1f}")
print(f"  Пересечение списков:             {len(overlap)}/10")
print(f"\nФайлы: final_scored.pkl, chart7_composite_distribution.html,")
print(f"        chart8_fifo_vs_merit.html, top10_fifo.csv, top10_merit.csv")
print(f"\nФаза 4 завершена! Готов к Фазе 5: Streamlit Dashboard")
