# =============================================================================
# ФАЗА 5: Streamlit Дашборд для комиссии МСХ Казахстана
# Система merit-based скоринга заявок на субсидии
# Запуск: streamlit run dashboard.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Увеличиваем лимит ячеек для Pandas Styler (по умолчанию 262144 — мало для 36k строк)
pd.set_option("styler.render.max_elements", 500_000)

# ----------------------------------------------------------------------------
# КОНФИГУРАЦИЯ СТРАНИЦЫ — должна быть первой командой Streamlit
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="АгроСкоринг — МСХ Казахстан",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Минималистичный CSS: убираем лишние отступы, стилизуем карточки KPI
st.markdown("""
<style>
    /* Основной фон */
    .main { background-color: #F8F9FA; }
    /* KPI-карточка */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #2E86C1;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #2E86C1; }
    .kpi-label { font-size: 0.85rem; color: #7F8C8D; margin-top: 4px; }
    /* Тир-бейджи */
    .tier-green  { background:#D5F5E3; color:#1E8449; padding:2px 10px; border-radius:12px; font-size:0.85rem; font-weight:600; }
    .tier-yellow { background:#FEF9E7; color:#B7950B; padding:2px 10px; border-radius:12px; font-size:0.85rem; font-weight:600; }
    .tier-red    { background:#FADBD8; color:#922B21; padding:2px 10px; border-radius:12px; font-size:0.85rem; font-weight:600; }
    /* Разделитель */
    hr { border: none; border-top: 1px solid #ECF0F1; margin: 16px 0; }
    /* Кнопки решений */
    div[data-testid="stHorizontalBlock"] button { border-radius: 8px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ЗАГРУЗКА ДАННЫХ — кешируем чтобы не перечитывать при каждом взаимодействии
# ----------------------------------------------------------------------------
FEATURE_COLS = [
    "log_amount", "log_heads", "amount_vs_region_avg",
    "heads_vs_direction_avg", "norm_vs_direction_avg",
    "region_approval_rate", "direction_approval_rate",
    "district_approval_rate", "region_avg_amount",
    "region_application_count", "month", "day_of_week",
    "hour", "is_early_month", "direction_enc",
    "region_enc", "district_enc", "subsidy_category_enc",
]

RULE_COLS = [
    "rule_min_heads", "rule_norm_positive", "rule_amount_positive",
    "rule_not_extreme_amount", "rule_norm_in_range",
]

RULE_LABELS = {
    "rule_min_heads":           "Поголовье ≥ 1",
    "rule_norm_positive":       "Норматив > 0",
    "rule_amount_positive":     "Сумма > 0",
    "rule_not_extreme_amount":  "Сумма ≤ 99-й перцентиль",
    "rule_norm_in_range":       "Норматив в диапазоне ±3σ",
}

@st.cache_data(show_spinner="Загрузка данных...")
def load_data():
    with open("final_scored.pkl", "rb") as f:
        df = pickle.load(f)
    return df

@st.cache_resource(show_spinner="Загрузка модели...")
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df_full = load_data()
model   = load_model()

# Инициализация session_state для хранения решений комиссии
if "decisions" not in st.session_state:
    st.session_state.decisions = {}   # {app_num: "Одобрено" | "Отклонено" | "Доп.данные"}
if "selected_row_idx" not in st.session_state:
    st.session_state.selected_row_idx = None

# ----------------------------------------------------------------------------
# САЙДБАР — фильтры
# ----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Flag_of_Kazakhstan.svg/320px-Flag_of_Kazakhstan.svg.png",
             width=60)
    st.title("🌾 АгроСкоринг")
    st.caption("Министерство сельского хозяйства РК")
    st.markdown("---")

    st.subheader("Фильтры")

    # Регион
    regions = ["Все регионы"] + sorted(df_full["region"].dropna().unique().tolist())
    sel_region = st.selectbox("🗺️ Регион", regions)

    # Направление животноводства
    directions = ["Все направления"] + sorted(df_full["direction"].dropna().unique().tolist())
    sel_direction = st.selectbox("🐄 Направление", directions)

    # Минимальный composite_score
    min_score = st.slider(
        "📊 Мин. Composite Score", min_value=0, max_value=100, value=0, step=5
    )

    # Фильтр по тиру
    tier_options = ["Все", "🟢 Приоритет", "🟡 Рассмотреть", "🔴 Отклонить"]
    sel_tier = st.radio("🏅 Тир", tier_options, horizontal=False)

    st.markdown("---")
    st.caption(f"Данные обновлены: {df_full['date'].max().date()}")
    st.caption(f"Всего заявок в БД: {len(df_full):,}")

# ----------------------------------------------------------------------------
# ПРИМЕНЯЕМ ФИЛЬТРЫ
# ----------------------------------------------------------------------------
df = df_full.copy()

if sel_region != "Все регионы":
    df = df[df["region"] == sel_region]
if sel_direction != "Все направления":
    df = df[df["direction"] == sel_direction]
df = df[df["composite_score"] >= min_score]
if sel_tier != "Все":
    df = df[df["tier"] == sel_tier]

df = df.reset_index(drop=True)

# ----------------------------------------------------------------------------
# ЗАГОЛОВОК
# ----------------------------------------------------------------------------
st.title("🌾 АгроСкоринг — Система Merit-Based приоритизации субсидий")
st.caption("Merit-based скоринг для объективной приоритизации. Финальное решение остаётся за комиссией.")

# ----------------------------------------------------------------------------
# БЛОК 1: KPI-КАРТОЧКИ
# ----------------------------------------------------------------------------
total_apps      = len(df)
priority_apps   = (df["tier"] == "🟢 Приоритет").sum()
avg_score       = df["composite_score"].mean() if total_apps > 0 else 0
total_amount    = df["amount"].sum() / 1_000_000_000  # в млрд тг

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card" style="border-color:#2E86C1">
      <div class="kpi-value">{total_apps:,}</div>
      <div class="kpi-label">Всего заявок в фильтре</div>
    </div>""", unsafe_allow_html=True)

with c2:
    pct = priority_apps / total_apps * 100 if total_apps > 0 else 0
    st.markdown(f"""
    <div class="kpi-card" style="border-color:#2ECC71">
      <div class="kpi-value" style="color:#1E8449">{priority_apps:,}</div>
      <div class="kpi-label">🟢 Приоритетных заявок ({pct:.1f}%)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card" style="border-color:#F39C12">
      <div class="kpi-value" style="color:#B7950B">{avg_score:.1f}</div>
      <div class="kpi-label">Средний Composite Score</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card" style="border-color:#9B59B6">
      <div class="kpi-value" style="color:#7D3C98">{total_amount:.1f} млрд</div>
      <div class="kpi-label">Сумма субсидий (тенге)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ВКЛАДКИ: Скоринглист | Аналитика
# ----------------------------------------------------------------------------
tab_list, tab_analytics = st.tabs(["📋 Скоринглист", "📊 Аналитика"])

# ============================================================================
# ВКЛАДКА 1: СКОРИНГЛИСТ
# ============================================================================
with tab_list:

    # ------------------------------------------------------------------------
    # БЛОК 2: ТАБЛИЦА SHORTLIST
    # ------------------------------------------------------------------------
    st.subheader("Приоритизированный список заявок")

    if total_apps == 0:
        st.warning("Нет заявок, соответствующих выбранным фильтрам.")
    else:
        # Ограничиваем таблицу до 500 строк для производительности браузера
        MAX_DISPLAY_ROWS = 500
        df_table = df.head(MAX_DISPLAY_ROWS).copy()
        if len(df) > MAX_DISPLAY_ROWS:
            st.caption(f"ℹ️ Показаны топ {MAX_DISPLAY_ROWS} заявок из {len(df):,}. Используйте фильтры для сужения выборки или экспортируйте в Excel.")

        # Готовим отображаемую таблицу
        def shorten_direction(d):
            return (str(d)
                    .replace("Субсидирование в ", "")
                    .replace("Субсидирование затрат по ", "")
                    .replace("Субсидирование ", "")
                    .strip().capitalize())

        df_display = pd.DataFrame({
            "Регион":        df_table["region"],
            "Район":         df_table["district"].fillna("—"),
            "Направление":   df_table["direction"].apply(shorten_direction),
            "Тип субсидии":  df_table["subsidy_category"],
            "Поголовье":     df_table["heads"].round(0).astype(int),
            "Сумма, тг":     df_table["amount"].apply(lambda x: f"{x:,.0f}"),
            "ML Score":      df_table["ml_score_100"].round(1),
            "Rule Score":    (df_table["rule_score"] * 100).round(1),
            "Composite":     df_table["composite_score"],
            "Тир":           df_table["tier"],
            "_idx":          df_table.index,   # скрытое поле для идентификации строки
        })

        # Цветовая кодировка строк по тиру
        def highlight_tier(row):
            tier = row["Тир"]
            if "Приоритет"   in tier: bg = "background-color: #eafaf1"
            elif "Рассмотреть" in tier: bg = "background-color: #fef9e7"
            else:                       bg = "background-color: #fdf2f2"
            return [bg] * len(row)

        styled = (
            df_display.drop(columns=["_idx"])
            .style.apply(highlight_tier, axis=1)
            .format({"Composite": "{:.1f}", "ML Score": "{:.1f}", "Rule Score": "{:.1f}"})
            .set_properties(**{"font-size": "0.85rem"})
        )

        st.dataframe(styled, use_container_width=True, height=380)

        # --- Кнопка экспорта в Excel ---
        col_exp, col_sel = st.columns([1, 3])
        with col_exp:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                df_display.drop(columns=["_idx"]).to_excel(
                    writer, index=False, sheet_name="Скоринглист"
                )
            st.download_button(
                label="📥 Экспорт в Excel",
                data=excel_buf.getvalue(),
                file_name=f"scoring_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col_sel:
            # Выбор строки для детального просмотра
            row_options = [f"#{i+1} | {df.iloc[i]['region']} | {shorten_direction(df.iloc[i]['direction'])} | {df.iloc[i]['composite_score']:.1f} баллов"
                           for i in range(min(len(df), 200))]
            selected_label = st.selectbox(
                "🔍 Выбрать заявку для детального анализа",
                options=["— выберите заявку —"] + row_options,
            )

        # ------------------------------------------------------------------------
        # БЛОК 3: ДЕТАЛЬНЫЙ ПРОСМОТР ЗАЯВКИ
        # ------------------------------------------------------------------------
        if selected_label and selected_label != "— выберите заявку —":
            row_num = int(selected_label.split("#")[1].split("|")[0].strip()) - 1
            row = df.iloc[row_num]

            st.markdown("---")
            st.subheader(f"📄 Детали заявки — {row['region']} | {shorten_direction(row['direction'])}")

            # Базовая информация о заявке
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Composite Score", f"{row['composite_score']:.1f}")
                st.metric("ML Score", f"{row['ml_score_100']:.1f}")
            with info_col2:
                st.metric("Сумма субсидии", f"{row['amount']/1_000_000:.2f} млн тг")
                st.metric("Поголовье (расчётное)", f"{row['heads']:.0f} голов")
            with info_col3:
                st.metric("Норматив", f"{row['norm']:,.0f} тг")
                tier_str = row["tier"]
                st.metric("Тир", tier_str)

            # --- Rule-based фильтры ---
            st.markdown("**✅ Прохождение правил:**")
            rule_cols_r = st.columns(len(RULE_COLS))
            for col_r, rule in zip(rule_cols_r, RULE_COLS):
                passed = int(row[rule]) == 1
                icon   = "✅" if passed else "❌"
                color  = "green" if passed else "red"
                col_r.markdown(
                    f"<span style='color:{color};font-size:1.3rem'>{icon}</span><br>"
                    f"<small>{RULE_LABELS[rule]}</small>",
                    unsafe_allow_html=True,
                )

            # --- Anomaly status ---
            is_anomaly = int(row.get("anomaly_label", 1)) == -1
            if is_anomaly:
                st.warning(f"⚠️ **Аномалия обнаружена** (Isolation Forest). "
                           f"Anomaly Score: {row.get('anomaly_score_norm', 0):.3f}. "
                           f"Рекомендуется дополнительная проверка.")
            else:
                st.success(f"✓ Заявка в пределах нормы (Anomaly Score: {row.get('anomaly_score_norm', 1):.3f})")

            # --- SHAP Waterfall для выбранной заявки ---
            st.markdown("**🔍 SHAP объяснение (почему такой скор):**")
            with st.spinner("Вычисляем SHAP для выбранной заявки..."):
                try:
                    X_row = row[FEATURE_COLS].values.reshape(1, -1)
                    X_row_df = pd.DataFrame(X_row, columns=FEATURE_COLS)
                    explainer = shap.TreeExplainer(model)
                    sv = explainer(X_row_df)

                    fig_wf, ax = plt.subplots(figsize=(9, 5))
                    shap.waterfall_plot(sv[0], max_display=10, show=False)
                    plt.title(f"SHAP — Composite Score: {row['composite_score']:.1f}",
                              fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig_wf, use_container_width=True)
                    plt.close()
                except Exception as e:
                    st.info(f"SHAP недоступен для этой строки: {e}")

            # --- Кнопки решений комиссии ---
            st.markdown("**⚖️ Решение комиссии (human-in-the-loop):**")

            app_key = str(row.get("app_num", row_num))
            current_decision = st.session_state.decisions.get(app_key)

            if current_decision:
                decision_colors = {
                    "Одобрено": "green", "Отклонено": "red", "Запрошены доп. данные": "orange"
                }
                color = decision_colors.get(current_decision, "grey")
                st.markdown(
                    f"<div style='padding:8px 16px;background:#F0F0F0;border-radius:8px;"
                    f"border-left:4px solid {color}'>"
                    f"Записанное решение: <b style='color:{color}'>{current_decision}</b>"
                    f"</div>", unsafe_allow_html=True
                )
                st.caption("Изменить решение:")

            btn1, btn2, btn3 = st.columns(3)
            with btn1:
                if st.button("✅ Одобрить", key=f"approve_{row_num}", type="primary"):
                    st.session_state.decisions[app_key] = "Одобрено"
                    st.success("✅ Решение записано: **Одобрено**")
                    st.rerun()
            with btn2:
                if st.button("❌ Отклонить", key=f"reject_{row_num}"):
                    st.session_state.decisions[app_key] = "Отклонено"
                    st.error("❌ Решение записано: **Отклонено**")
                    st.rerun()
            with btn3:
                if st.button("📋 Запросить доп. данные", key=f"more_{row_num}"):
                    st.session_state.decisions[app_key] = "Запрошены доп. данные"
                    st.warning("📋 Решение записано: **Запрошены дополнительные данные**")
                    st.rerun()

            # Счётчик принятых решений в сессии
            if st.session_state.decisions:
                with st.expander(f"📊 Решения комиссии в этой сессии ({len(st.session_state.decisions)} шт.)"):
                    decisions_df = pd.DataFrame(
                        [(k, v) for k, v in st.session_state.decisions.items()],
                        columns=["Номер заявки", "Решение"]
                    )
                    st.dataframe(decisions_df, use_container_width=True)
                    if st.button("Очистить все решения"):
                        st.session_state.decisions = {}
                        st.rerun()

# ============================================================================
# ВКЛАДКА 2: АНАЛИТИКА
# ============================================================================
with tab_analytics:
    st.subheader("📊 Аналитика по отфильтрованным заявкам")

    if total_apps == 0:
        st.warning("Нет данных для аналитики. Измените фильтры.")
    else:
        # --- График 1: Распределение composite_score ---
        ana_c1, ana_c2 = st.columns(2)

        with ana_c1:
            st.markdown("#### Распределение Composite Score")
            fig_hist = go.Figure()
            tier_palette = {
                "🟢 Приоритет":   "#2ECC71",
                "🟡 Рассмотреть": "#F39C12",
                "🔴 Отклонить":   "#E74C3C",
            }
            for tier_name, color in tier_palette.items():
                subset = df[df["tier"] == tier_name]["composite_score"]
                if len(subset) > 0:
                    fig_hist.add_trace(go.Histogram(
                        x=subset, name=tier_name,
                        marker_color=color, opacity=0.75, nbinsx=30,
                    ))
            fig_hist.add_vline(x=75, line_dash="dash", line_color="#2ECC71",
                               annotation_text="≥75 Приоритет")
            fig_hist.add_vline(x=50, line_dash="dash", line_color="#F39C12",
                               annotation_text="≥50 Рассмотреть")
            fig_hist.update_layout(
                barmode="overlay",
                xaxis_title="Composite Score",
                yaxis_title="Количество заявок",
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(title="Тир"),
                height=340, margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # --- График 2: Scatter: сумма vs score ---
        with ana_c2:
            st.markdown("#### Сумма субсидии vs Composite Score")
            df_plot = df.copy()
            df_plot["amount_mln"] = df_plot["amount"] / 1_000_000
            # Срезаем выбросы для читаемости
            p99_a = df_plot["amount_mln"].quantile(0.99)
            df_plot = df_plot[df_plot["amount_mln"] <= p99_a]

            fig_sc = px.scatter(
                df_plot,
                x="composite_score",
                y="amount_mln",
                color="tier",
                color_discrete_map={
                    "🟢 Приоритет":   "#2ECC71",
                    "🟡 Рассмотреть": "#F39C12",
                    "🔴 Отклонить":   "#E74C3C",
                },
                opacity=0.5,
                labels={
                    "composite_score": "Composite Score",
                    "amount_mln":      "Сумма субсидии (млн тг)",
                    "tier":            "Тир",
                },
                hover_data=["region", "direction"],
            )
            fig_sc.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=340, margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # --- График 3: Средний score по регионам ---
        st.markdown("#### Средний Composite Score по регионам")
        region_scores = (
            df.groupby("region")["composite_score"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_score", "count": "n_apps"})
            .sort_values("avg_score", ascending=True)
        )

        fig_reg = go.Figure(go.Bar(
            x=region_scores["avg_score"],
            y=region_scores["region"],
            orientation="h",
            marker_color=px.colors.sequential.Blues[3:][
                ::-1][:len(region_scores)] if len(region_scores) <= 9
                else ["#2E86C1"] * len(region_scores),
            text=region_scores.apply(
                lambda r: f"{r['avg_score']:.1f} ({r['n_apps']:,} заяв.)", axis=1),
            textposition="outside",
        ))
        fig_reg.update_layout(
            xaxis_title="Средний Composite Score",
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(300, len(region_scores) * 30 + 60),
            margin=dict(t=20, l=220, r=80),
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # --- Таблица: FIFO vs Merit-based ---
        st.markdown("#### ⚡ Старая система (FIFO) vs Новая (merit-based)")
        st.caption("Топ-10 заявок по дате подачи против топ-10 по скору")

        def shorten_dir(d):
            return (str(d)
                    .replace("Субсидирование в ", "")
                    .replace("Субсидирование ", "")
                    .strip().capitalize())

        top10_fifo = (
            df.sort_values("date").head(10)
            [["region", "direction", "amount", "composite_score", "tier", "date"]]
            .copy()
        )
        top10_fifo["Метод"] = "📅 FIFO (по дате)"
        top10_fifo["Ранг"] = range(1, len(top10_fifo)+1)

        top10_merit = (
            df.sort_values("composite_score", ascending=False).head(10)
            [["region", "direction", "amount", "composite_score", "tier", "date"]]
            .copy()
        )
        top10_merit["Метод"] = "⭐ Merit (по скору)"
        top10_merit["Ранг"] = range(1, len(top10_merit)+1)

        combined = pd.concat([top10_fifo, top10_merit], ignore_index=True)
        combined["amount_mln"] = (combined["amount"] / 1_000_000).round(2)
        combined["direction_s"] = combined["direction"].apply(shorten_dir)

        display_combined = combined[[
            "Метод", "Ранг", "region", "direction_s",
            "amount_mln", "composite_score", "tier"
        ]].rename(columns={
            "region":       "Регион",
            "direction_s":  "Направление",
            "amount_mln":   "Сумма, млн тг",
            "composite_score": "Composite Score",
            "tier":         "Тир",
        })

        def highlight_method(row):
            if "FIFO" in row["Метод"]:
                return ["background-color: #EBF5FB"] * len(row)
            return ["background-color: #EAFAF1"] * len(row)

        st.dataframe(
            display_combined.style.apply(highlight_method, axis=1)
            .format({"Composite Score": "{:.1f}", "Сумма, млн тг": "{:.2f}"}),
            use_container_width=True,
            height=460,
        )

        # Ключевой вывод
        avg_fifo  = top10_fifo["composite_score"].mean()
        avg_merit = top10_merit["composite_score"].mean()
        delta     = avg_merit - avg_fifo
        overlap   = len(
            set(top10_fifo.index.tolist()) & set(top10_merit.index.tolist())
        )

        metric_c1, metric_c2, metric_c3 = st.columns(3)
        metric_c1.metric("Средний скор FIFO топ-10",  f"{avg_fifo:.1f}")
        metric_c2.metric("Средний скор Merit топ-10", f"{avg_merit:.1f}", delta=f"+{delta:.1f}")
        metric_c3.metric(
            "Пересечение списков",
            f"{overlap}/10",
            delta=f"{10 - overlap} новых заявок",
            delta_color="normal",
        )

        st.info(
            f"💡 **Вывод:** При merit-based подходе средний скор топ-10 выше на **{delta:.1f} балла**. "
            f"**{10 - overlap} из 10** заявок в приоритетном списке меняются по сравнению с FIFO. "
            f"Система объективно находит более достойные заявки независимо от времени подачи."
        )
