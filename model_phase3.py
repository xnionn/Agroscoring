# =============================================================================
# ФАЗА 3: ML модель (XGBoost) + SHAP объяснения
# Система скоринга заявок на сельскохозяйственные субсидии — МСХ Казахстана
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib
matplotlib.use("Agg")  # без GUI — для сохранения SHAP-графиков в файл
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# ШАГ 1: ПОДГОТОВКА ДАННЫХ
# ----------------------------------------------------------------------------
print("=" * 60)
print("ШАГ 1: Загрузка и подготовка данных...")
print("=" * 60)

with open("features_data.pkl", "rb") as f:
    df = pickle.load(f)

print(f"Загружено: {df.shape[0]:,} строк × {df.shape[1]} колонок")

# Оставляем только завершённые заявки (target не NaN)
df_model = df[df["target"].notna()].copy()
df_model["target"] = df_model["target"].astype(int)

print(f"После фильтрации (Исполнена + Отклонена): {len(df_model):,} строк\n")

# Полный список признаков — те же 18, что созданы в Фазе 2
FEATURE_COLS = [
    "log_amount",
    "log_heads",
    "amount_vs_region_avg",
    "heads_vs_direction_avg",
    "norm_vs_direction_avg",
    "region_approval_rate",
    "direction_approval_rate",
    "district_approval_rate",
    "region_avg_amount",
    "region_application_count",
    "month",
    "day_of_week",
    "hour",
    "is_early_month",
    "direction_enc",
    "region_enc",
    "district_enc",
    "subsidy_category_enc",
]

X = df_model[FEATURE_COLS].values
y = df_model["target"].values

# Стратифицированный сплит 80/20 (сохраняем баланс классов в обоих сплитах)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {len(X_train):,} строк  |  Test: {len(X_test):,} строк")
print(f"\nБаланс классов в Train:")
print(f"  y=1 (Исполнена): {y_train.sum():,}  ({y_train.mean()*100:.1f}%)")
print(f"  y=0 (Отклонена): {(y_train==0).sum():,}  ({(y_train==0).mean()*100:.1f}%)")
print(f"\nБаланс классов в Test:")
print(f"  y=1 (Исполнена): {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
print(f"  y=0 (Отклонена): {(y_test==0).sum():,}  ({(y_test==0).mean()*100:.1f}%)")

# ----------------------------------------------------------------------------
# ШАГ 2: BASELINE — Logistic Regression
# Точка отсчёта: простая линейная модель без тюнинга
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 2: Baseline — Logistic Regression...")
print("=" * 60)

# Масштабирование обязательно для LogReg (градиентный спуск чувствителен к масштабу)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(
    class_weight="balanced",  # автоматически компенсирует дисбаланс классов
    max_iter=1000,
    random_state=42,
)
lr.fit(X_train_sc, y_train)

y_pred_lr = lr.predict(X_test_sc)
y_proba_lr = lr.predict_proba(X_test_sc)[:, 1]
auc_lr = roc_auc_score(y_test, y_proba_lr)

print("\nLogistic Regression — classification report:")
print(classification_report(y_test, y_pred_lr,
      target_names=["Отклонена (0)", "Исполнена (1)"]))
print(f"ROC-AUC (Logistic Regression): {auc_lr:.4f}")

# ----------------------------------------------------------------------------
# ШАГ 3: ОСНОВНАЯ МОДЕЛЬ — XGBoostClassifier
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 3: XGBoost Classifier...")
print("=" * 60)

# scale_pos_weight: компенсирует дисбаланс классов
# = count(отрицательных) / count(положительных) ≈ 0.14
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw   = n_neg / n_pos
print(f"scale_pos_weight = {n_neg} / {n_pos} = {spw:.4f}\n")

xgb = XGBClassifier(
    n_estimators      = 300,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = spw,
    random_state      = 42,
    eval_metric       = "auc",       # метрика для early stopping
    early_stopping_rounds = 20,      # остановимся если AUC не растёт 20 раундов
    verbosity         = 0,
)

# eval_set позволяет отслеживать метрику на валидации и останавливать раньше
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

best_iter = xgb.best_iteration
print(f"Лучшая итерация (early stopping): {best_iter}")

y_pred_xgb   = xgb.predict(X_test)
y_proba_xgb  = xgb.predict_proba(X_test)[:, 1]
auc_xgb      = roc_auc_score(y_test, y_proba_xgb)

print("\nXGBoost — classification report:")
print(classification_report(y_test, y_pred_xgb,
      target_names=["Отклонена (0)", "Исполнена (1)"]))
print(f"ROC-AUC (XGBoost):              {auc_xgb:.4f}")
print(f"ROC-AUC (Logistic Regression):  {auc_lr:.4f}")
print(f"Прирост AUC vs Baseline:        +{auc_xgb - auc_lr:.4f}")

# --- 3.1 Plotly: Feature Importance топ-15 ---
print("\nПостроение графика Feature Importance...")

importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLS)
top15_imp   = importances.sort_values(ascending=False).head(15)
top15_imp   = top15_imp.sort_values(ascending=True)  # горизонтальный бар

# Цветовая шкала: чем важнее — тем насыщеннее
colors = px.colors.sequential.Blues[2:]
color_list = [
    colors[int(i / len(top15_imp) * (len(colors) - 1))]
    for i in range(len(top15_imp))
]

fig_imp = go.Figure(go.Bar(
    x=top15_imp.values,
    y=top15_imp.index,
    orientation="h",
    marker_color=color_list,
    text=[f"{v:.4f}" for v in top15_imp.values],
    textposition="outside",
))
fig_imp.update_layout(
    title=f"XGBoost — топ-15 важных признаков (ROC-AUC: {auc_xgb:.4f})",
    xaxis_title="Feature Importance (gain)",
    yaxis_title="Признак",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=12),
    height=500,
    margin=dict(t=60, l=220, r=80),
)
fig_imp.write_html("chart6_feature_importance.html")
print("Сохранено: chart6_feature_importance.html")

# ----------------------------------------------------------------------------
# ШАГ 4: SHAP ОБЪЯСНЕНИЯ
# TreeExplainer оптимизирован для tree-based моделей (быстрее KernelExplainer)
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 4: SHAP объяснения...")
print("=" * 60)

# Создаём explainer на тестовой выборке
# X_test_df — датафрейм с именами колонок (нужен для SHAP-графиков)
X_test_df  = pd.DataFrame(X_test, columns=FEATURE_COLS)
X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)

explainer   = shap.TreeExplainer(xgb)
shap_values = explainer(X_test_df)  # объект shap.Explanation

print(f"SHAP values shape: {shap_values.values.shape}")

# --- 4.1 Summary plot (beeswarm) — обзор влияния всех признаков ---
print("  → SHAP Summary plot (beeswarm)...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values.values,
    X_test_df,
    feature_names=FEATURE_COLS,
    show=False,
    plot_size=(10, 7),
)
plt.title("SHAP Summary — влияние признаков на предсказание", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("     Сохранено: shap_summary.png")

# --- 4.2 Выбираем 3 примера для Waterfall plots ---
# Вычисляем скоры для всей тестовой выборки
scores_test = y_proba_xgb

idx_best     = int(np.argmax(scores_test))           # наивысший скор
idx_worst    = int(np.argmin(scores_test))            # наименьший скор
idx_border   = int(np.argmin(np.abs(scores_test - 0.5)))  # ближайший к 0.5

print(f"\n  Лучшая заявка  (idx={idx_best}):     score = {scores_test[idx_best]:.4f}")
print(f"  Худшая заявка  (idx={idx_worst}):    score = {scores_test[idx_worst]:.4f}")
print(f"  Пограничная    (idx={idx_border}):   score = {scores_test[idx_border]:.4f}")

# --- 4.3 Waterfall plots для каждого из 3 примеров ---
for label, idx in [("best", idx_best), ("worst", idx_worst), ("border", idx_border)]:
    score  = scores_test[idx]
    actual = y_test[idx]
    title_map = {
        "best":   f"SHAP Waterfall — «Лучшая заявка» (score={score:.3f}, факт={'Исполнена' if actual==1 else 'Отклонена'})",
        "worst":  f"SHAP Waterfall — «Худшая заявка» (score={score:.3f}, факт={'Исполнена' if actual==1 else 'Отклонена'})",
        "border": f"SHAP Waterfall — «Пограничная заявка» (score={score:.3f}, факт={'Исполнена' if actual==1 else 'Отклонена'})",
    }
    fig_wf, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(shap_values[idx], max_display=12, show=False)
    plt.title(title_map[label], fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Сохранено: shap_waterfall_{label}.png")

# --- 4.4 Сохраняем SHAP values ---
with open("shap_values.pkl", "wb") as f:
    pickle.dump({
        "shap_values":   shap_values.values,     # numpy array
        "base_value":    shap_values.base_values[0],
        "feature_names": FEATURE_COLS,
        "X_test":        X_test,
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print("\nСохранено: shap_values.pkl")

# ----------------------------------------------------------------------------
# ШАГ 5: SCORING — добавляем ml_score в полный датафрейм
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 5: Скоринг всего датафрейма...")
print("=" * 60)

# Применяем модель ко ВСЕМ заявкам (включая Одобрена, Отозвана и т.д.)
# Это ключевая ценность системы — скорировать весь поток заявок
X_all = df[FEATURE_COLS].values
df["ml_score"]     = xgb.predict_proba(X_all)[:, 1]        # вероятность (0–1)
df["ml_score_100"] = (df["ml_score"] * 100).round(1)       # нормировано (0–100)

# Добавляем категорию риска для удобства комиссии
def score_to_band(s):
    if s >= 80:   return "🟢 Высокий (≥80)"
    if s >= 60:   return "🟡 Средний (60–79)"
    if s >= 40:   return "🟠 Пониженный (40–59)"
    return              "🔴 Низкий (<40)"

df["score_band"] = df["ml_score_100"].apply(score_to_band)

print(f"Скоры добавлены для {len(df):,} заявок")
print(f"\nРаспределение по бэндам:")
print(df["score_band"].value_counts().to_string())

print(f"\nСтатистика ml_score_100:")
desc = df["ml_score_100"].describe()
print(desc.apply(lambda x: f"{x:.1f}").to_string())

# Сохраняем модель
with open("model.pkl", "wb") as f:
    pickle.dump(xgb, f, protocol=pickle.HIGHEST_PROTOCOL)
print("\nСохранено: model.pkl")

# Сохраняем полный датафрейм со скорами
with open("scored_data.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Сохранено: scored_data.pkl")

# ----------------------------------------------------------------------------
# ФИНАЛЬНАЯ СВОДКА ФАЗЫ 3
# ----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ФИНАЛЬНАЯ СВОДКА ФАЗЫ 3")
print("=" * 60)
print(f"Обучающая выборка:           {len(X_train):,} строк")
print(f"Тестовая выборка:            {len(X_test):,} строк")
print(f"")
print(f"Baseline (Logistic Reg.):")
print(f"  ROC-AUC:                   {auc_lr:.4f}")
print(f"")
print(f"XGBoost (best_iter={best_iter}):")
print(f"  ROC-AUC:                   {auc_xgb:.4f}  (+{auc_xgb-auc_lr:.4f} vs baseline)")
print(f"")
print(f"Топ-3 самых важных признака:")
for feat, val in importances.sort_values(ascending=False).head(3).items():
    print(f"  {feat:<35s} {val:.4f}")
print(f"")
print(f"Файлы: model.pkl, scored_data.pkl, shap_values.pkl")
print(f"       chart6_feature_importance.html")
print(f"       shap_summary.png, shap_waterfall_*.png")
print(f"\nФаза 3 завершена успешно! Готов к Фазе 4: Streamlit Dashboard")
