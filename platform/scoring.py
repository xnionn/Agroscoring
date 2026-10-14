"""
ML Scoring Service — loads trained model and scores new applications in real-time.
Uses model.pkl, encoders.pkl, and cached benchmarks from features_data.pkl.
Integrates real datasets from Downloads for enhanced regional benchmarking.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import shap
from datetime import datetime, timezone
from dataset_service import dataset_service

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Feature columns - same as training
FEATURE_COLS = [
    "log_amount", "log_heads", "amount_vs_region_avg",
    "heads_vs_direction_avg", "norm_vs_direction_avg",
    "region_approval_rate", "direction_approval_rate", "district_approval_rate",
    "region_avg_amount", "region_application_count",
    "month", "day_of_week", "hour", "is_early_month",
    "direction_enc", "region_enc", "district_enc", "subsidy_category_enc",
]

FEATURE_LABELS_RU = {
    "log_amount": "Сумма субсидии (лог.)",
    "log_heads": "Поголовье (лог.)",
    "amount_vs_region_avg": "Сумма vs среднее по региону",
    "heads_vs_direction_avg": "Поголовье vs среднее по направлению",
    "norm_vs_direction_avg": "Норматив vs среднее по направлению",
    "region_approval_rate": "% одобрения в регионе",
    "direction_approval_rate": "% одобрения по направлению",
    "district_approval_rate": "% одобрения в районе",
    "region_avg_amount": "Средняя сумма в регионе",
    "region_application_count": "Всего заявок из региона",
    "month": "Месяц подачи",
    "day_of_week": "День недели",
    "hour": "Час подачи",
    "is_early_month": "Начало месяца (до 10-го)",
    "direction_enc": "Направление животноводства",
    "region_enc": "Регион",
    "district_enc": "Район",
    "subsidy_category_enc": "Категория субсидии",
}

# Anomaly detection features
ANOMALY_FEATURES = ["log_amount", "log_heads", "norm_vs_direction_avg", "amount_vs_region_avg"]


class ScoringService:
    def __init__(self):
        self.model = None
        self.encoders = None
        self.explainer = None
        self.benchmarks = {}
        self.iso_forest = None
        self.sources = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        # Load model
        with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        # Load encoders
        with open(os.path.join(BASE_DIR, "encoders.pkl"), "rb") as f:
            self.encoders = pickle.load(f)
        # Load features data for benchmarks
        with open(os.path.join(BASE_DIR, "features_data.pkl"), "rb") as f:
            df = pickle.load(f)

        # Load real datasets from Downloads for enhanced benchmarking
        try:
            dataset_service.load()
        except Exception as e:
            print(f"[ScoringService] Warning: Could not load external datasets: {e}")

        # Cache benchmarks from historical data
        completed = df[df["status"].isin(["Исполнена", "Отклонена"])].copy()
        completed["target"] = (completed["status"] == "Исполнена").astype(int)

        self.benchmarks["region_approval_rate"] = completed.groupby("region")["target"].mean().to_dict()
        self.benchmarks["direction_approval_rate"] = completed.groupby("direction")["target"].mean().to_dict()
        self.benchmarks["district_approval_rate"] = completed.groupby("district")["target"].mean().to_dict()
        self.benchmarks["region_avg_amount"] = df.groupby("region")["amount"].mean().to_dict()
        self.benchmarks["region_application_count"] = completed.groupby("region")["target"].count().to_dict()
        self.benchmarks["direction_mean_heads"] = df.groupby("direction")["heads"].mean().to_dict()
        self.benchmarks["direction_mean_norm"] = df.groupby("direction")["norm"].mean().to_dict()
        self.benchmarks["region_mean_amount"] = df.groupby("region")["amount"].mean().to_dict()

        # P99 amounts per direction for rule check
        self.benchmarks["p99_amount_by_direction"] = df.groupby("direction")["amount"].quantile(0.99).to_dict()
        # Norm stats per direction
        self.benchmarks["norm_mean_by_direction"] = df.groupby("direction")["norm"].mean().to_dict()
        self.benchmarks["norm_std_by_direction"] = df.groupby("direction")["norm"].std().fillna(0).to_dict()

        # Median benchmark values for unknown lookups
        self.benchmarks["median_district_rate"] = completed.groupby("district")["target"].mean().median()
        self.benchmarks["median_region_rate"] = completed.groupby("region")["target"].mean().median()
        self.benchmarks["median_direction_rate"] = completed.groupby("direction")["target"].mean().median()

        # SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Load or fit isolation forest
        try:
            from sklearn.ensemble import IsolationForest
            X_anom = df[ANOMALY_FEATURES].values
            self.iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
            self.iso_forest.fit(X_anom)
        except Exception:
            self.iso_forest = None

        self.sources = self._load_sources()

        self._loaded = True
        print("[ScoringService] Model, encoders, benchmarks loaded OK")

    def _load_sources(self) -> list:
        default_sources = [
            {
                "id": "hist_features",
                "title": "Исторические данные заявок (внутренний датасет проекта)",
                "url": "local:features_data.pkl",
                "publisher": "Decent Coders",
                "used_for": ["ml", "rules", "benchmarks"],
            },
            {
                "id": "constitution_kz",
                "title": "Конституция Республики Казахстан",
                "url": "https://adilet.zan.kz/rus/docs/K950001000_",
                "publisher": "Әділет (ИПС НПА РК)",
                "used_for": ["legal_principles"],
            },
            {
                "id": "stat_livestock",
                "title": "Статистика сельского, лесного, охотничьего и рыбного хозяйства",
                "url": "https://stat.gov.kz/ru/industries/businessstatistics/stat-forrest-village-hunt-fish/",
                "publisher": "Бюро национальной статистики РК",
                "used_for": ["regional_benchmarks"],
            },
            {
                "id": "egov_subsidy_sko",
                "title": "Выплаченные субсидии по развитию племенного животноводства (СКО)",
                "url": "https://data.egov.kz/datasets/view?index=subsidiizhivotnovodstvosko1",
                "publisher": "data.egov.kz",
                "used_for": ["regional_capacity_reference"],
            },
            {
                "id": "kazhydromet_meteo",
                "title": "Метеорологическая база данных",
                "url": "https://meteo.kazhydromet.kz/database_meteo/",
                "publisher": "РГП Казгидромет",
                "used_for": ["climate_risk_reference"],
            },
            {
                "id": "real_subsidy_data_2025",
                "title": "Реальные выданные субсидии 2025 (деанонимизированная выгрузка системы)",
                "url": "local:Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx",
                "publisher": "Система AgroSubsidia",
                "used_for": ["regional_benchmarks", "norm_validation"],
            },
            {
                "id": "facility_registry_production",
                "title": "Реестр производителей и переработчиков",
                "url": "local:reestr_uboihraneniepererab-v10.json",
                "publisher": "e-Government РК (data.egov.kz)",
                "used_for": ["facility_verification", "regional_context"],
            },
            {
                "id": "farmer_registry_family_farms",
                "title": "Реестр крестьянского хозяйства и семейных ферм",
                "url": "local:shko_sharua_zane_fermerlik_ko.csv",
                "publisher": "e-Government РК (data.egov.kz)",
                "used_for": ["farmer_context"],
            },
        ]

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scoring_sources.json")
        if not os.path.exists(path):
            return default_sources
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
        return default_sources

    def classify_subsidy(self, name: str) -> str:
        if not name:
            return "прочее"
        n = str(name).lower()
        if "осеменени" in n or "семя" in n:
            return "осеменение"
        if "приобретени" in n or "приобретен" in n:
            return "приобретение"
        if "удешевлени" in n:
            return "удешевление"
        if "племенной работ" in n or "селекционной и племенной" in n:
            return "племенная_работа"
        return "прочее"

    def _safe_encode(self, encoder_key: str, value: str) -> int:
        le = self.encoders.get(encoder_key)
        if le is None:
            return 0
        val_str = str(value) if value else "Unknown"
        if val_str in le.classes_:
            return int(le.transform([val_str])[0])
        return int(le.transform(["Unknown"])[0]) if "Unknown" in le.classes_ else 0

    def build_features(self, data: dict) -> np.ndarray:
        amount = float(data.get("amount", 0))
        norm = float(data.get("norm", 1))
        heads = amount / norm if norm > 0 else 0
        region = data.get("region", "")
        district = data.get("district", "")
        direction = data.get("direction", "")
        subsidy_name = data.get("subsidy_name", "")
        subsidy_category = self.classify_subsidy(subsidy_name)

        from datetime import datetime
        now = datetime.now()

        # Numeric features
        log_amount = np.log1p(amount)
        log_heads = np.log1p(heads)
        region_mean_amt = self.benchmarks["region_mean_amount"].get(region, 1)
        amount_vs_region_avg = amount / region_mean_amt if region_mean_amt > 0 else 1.0
        dir_mean_heads = self.benchmarks["direction_mean_heads"].get(direction, 1)
        heads_vs_direction_avg = heads / dir_mean_heads if dir_mean_heads > 0 else 1.0
        dir_mean_norm = self.benchmarks["direction_mean_norm"].get(direction, 1)
        norm_vs_direction_avg = norm / dir_mean_norm if dir_mean_norm > 0 else 1.0

        # Benchmarks
        region_approval_rate = self.benchmarks["region_approval_rate"].get(
            region, self.benchmarks["median_region_rate"])
        direction_approval_rate = self.benchmarks["direction_approval_rate"].get(
            direction, self.benchmarks["median_direction_rate"])
        district_approval_rate = self.benchmarks["district_approval_rate"].get(
            district, self.benchmarks["median_district_rate"])
        region_avg_amount = self.benchmarks["region_avg_amount"].get(region, amount)
        region_application_count = self.benchmarks["region_application_count"].get(region, 0)

        # Time features
        month = now.month
        day_of_week = now.weekday()
        hour = now.hour
        is_early_month = 1 if now.day <= 10 else 0

        # Encoded categories
        direction_enc = self._safe_encode("direction", direction)
        region_enc = self._safe_encode("region", region)
        district_enc = self._safe_encode("district", district)
        subsidy_category_enc = self._safe_encode("subsidy_category", subsidy_category)

        features = np.array([
            log_amount, log_heads, amount_vs_region_avg,
            heads_vs_direction_avg, norm_vs_direction_avg,
            region_approval_rate, direction_approval_rate, district_approval_rate,
            region_avg_amount, region_application_count,
            month, day_of_week, hour, is_early_month,
            direction_enc, region_enc, district_enc, subsidy_category_enc,
        ]).reshape(1, -1)

        return features

    def check_rules(self, data: dict) -> dict:
        amount = float(data.get("amount", 0))
        norm = float(data.get("norm", 1))
        heads = amount / norm if norm > 0 else 0
        direction = data.get("direction", "")
        profile_heads = data.get("profile_heads_count")
        profile_years = data.get("profile_years_active")
        profile_land = data.get("profile_land_area_ha")
        profile_revenue = data.get("profile_annual_revenue")
        profile_employees = data.get("profile_employee_count")
        lease_years_left = data.get("profile_lease_term_remaining_years")
        own_contribution_percent = data.get("own_contribution_percent")
        planned_execution_months = data.get("planned_execution_months")
        has_supporting_documents = data.get("has_supporting_documents")
        previous_subsidies_count = data.get("previous_subsidies_count")
        previous_returns_count = data.get("previous_subsidy_returns_count")
        previous_violations_count = data.get("previous_subsidy_violations_count")
        previous_overdue_reports_count = data.get("previous_subsidy_overdue_reports_count")
        previous_closed_properly = data.get("previous_subsidies_closed_properly")
        requested_purpose = str(data.get("requested_purpose", "") or "").strip()

        infra_flags = [
            data.get("profile_has_livestock_facility"),
            data.get("profile_has_feed_base"),
            data.get("profile_has_equipment"),
            data.get("profile_has_sales_contracts"),
        ]
        finance_flags = [
            data.get("profile_has_tax_debt"),
            data.get("profile_has_overdue_loans"),
        ]

        r1 = int(heads >= 1)
        r2 = int(norm > 0)
        r3 = int(amount > 0)

        p99 = self.benchmarks["p99_amount_by_direction"].get(direction, float("inf"))
        r4 = int(amount <= p99)

        n_mean = self.benchmarks["norm_mean_by_direction"].get(direction, norm)
        n_std = self.benchmarks["norm_std_by_direction"].get(direction, 0)
        r5 = int((norm >= n_mean - 3 * n_std) and (norm <= n_mean + 3 * n_std))

        # Cross-check with real dataset norms if available
        try:
            real_norm_check = dataset_service.validate_norm_reasonableness(direction, norm)
            if not real_norm_check.get("valid", True):
                # If real data shows norm is unreasonable, downgrade r5
                r5 = max(0, r5 - 1)
        except Exception:
            pass

        # Profile-aware criteria
        exp_score = 0.5
        if profile_years is not None:
            try:
                exp_score = min(max(float(profile_years), 0.0) / 10.0, 1.0)
            except Exception:
                exp_score = 0.5

        capacity_score = 0.5
        if profile_heads is not None:
            try:
                profile_heads_f = max(float(profile_heads), 0.0)
                required_heads = max(heads, 1e-6)
                capacity_score = min(profile_heads_f / required_heads, 1.0)
            except Exception:
                capacity_score = 0.5

        land_score = 0.5
        if profile_land is not None:
            try:
                land_f = max(float(profile_land), 0.0)
                heads_per_ha = heads / land_f if land_f > 0 else float("inf")
                # Conservative default capacity proxy for livestock intensity
                land_score = 1.0 if heads_per_ha <= 10 else max(0.0, 1.0 - (heads_per_ha - 10) / 20)
            except Exception:
                land_score = 0.5

        revenue_score = 0.5
        if profile_revenue is not None:
            try:
                revenue_f = max(float(profile_revenue), 0.0)
                if revenue_f <= 0:
                    revenue_score = 0.1
                else:
                    revenue_ratio = amount / revenue_f
                    if revenue_ratio <= 0.5:
                        revenue_score = 1.0
                    elif revenue_ratio <= 1.0:
                        revenue_score = 0.85
                    elif revenue_ratio <= 2.0:
                        revenue_score = 0.6
                    elif revenue_ratio <= 3.0:
                        revenue_score = 0.35
                    else:
                        revenue_score = 0.1
            except Exception:
                revenue_score = 0.5

        workforce_score = 0.5
        if profile_employees is not None:
            try:
                employee_count_f = max(float(profile_employees), 0.0)
                if employee_count_f <= 0:
                    workforce_score = 0.2
                else:
                    heads_per_employee = heads / employee_count_f if heads > 0 else 0
                    if heads_per_employee <= 25:
                        workforce_score = 1.0
                    elif heads_per_employee <= 60:
                        workforce_score = 0.7
                    elif heads_per_employee <= 100:
                        workforce_score = 0.4
                    else:
                        workforce_score = 0.2
            except Exception:
                workforce_score = 0.5

        infrastructure_score = 0.5
        provided_infra_flags = [flag for flag in infra_flags if flag is not None]
        if provided_infra_flags:
            infrastructure_score = float(np.mean([1.0 if flag else 0.0 for flag in provided_infra_flags]))

        financial_discipline_score = 0.5
        provided_finance_flags = [flag for flag in finance_flags if flag is not None]
        if provided_finance_flags:
            penalties = sum(1 for flag in provided_finance_flags if flag)
            financial_discipline_score = max(0.0, 1.0 - penalties * 0.5)

        cofinancing_score = 0.5
        if own_contribution_percent is not None:
            try:
                cofinancing_score = min(max(float(own_contribution_percent), 0.0) / 30.0, 1.0)
            except Exception:
                cofinancing_score = 0.5

        documentation_score = 0.5
        if has_supporting_documents is not None:
            documentation_score = 1.0 if has_supporting_documents else 0.2

        subsidy_history_score = 0.5
        history_values = [
            previous_subsidies_count,
            previous_returns_count,
            previous_violations_count,
            previous_overdue_reports_count,
            previous_closed_properly,
        ]
        if any(value is not None for value in history_values):
            try:
                prev_count = max(int(previous_subsidies_count or 0), 0)
                returns_count = max(int(previous_returns_count or 0), 0)
                violations_count = max(int(previous_violations_count or 0), 0)
                overdue_reports_count = max(int(previous_overdue_reports_count or 0), 0)
                if prev_count == 0:
                    subsidy_history_score = 0.7
                else:
                    subsidy_history_score = 1.0
                subsidy_history_score -= min(returns_count * 0.25, 0.5)
                subsidy_history_score -= min(violations_count * 0.3, 0.6)
                subsidy_history_score -= min(overdue_reports_count * 0.15, 0.45)
                if prev_count > 0 and previous_closed_properly is False:
                    subsidy_history_score -= 0.35
                subsidy_history_score = min(max(subsidy_history_score, 0.0), 1.0)
            except Exception:
                subsidy_history_score = 0.5

        execution_score = 0.5
        if planned_execution_months is not None:
            try:
                exec_months = max(int(planned_execution_months), 0)
                if exec_months <= 12:
                    execution_score = 1.0
                elif exec_months <= 18:
                    execution_score = 0.8
                elif exec_months <= 24:
                    execution_score = 0.6
                else:
                    execution_score = 0.35
            except Exception:
                execution_score = 0.5

        lease_security_score = 0.5
        if lease_years_left is not None:
            try:
                years_left = max(float(lease_years_left), 0.0)
                if years_left >= 5:
                    lease_security_score = 1.0
                elif years_left >= 3:
                    lease_security_score = 0.75
                elif years_left >= 1:
                    lease_security_score = 0.5
                else:
                    lease_security_score = 0.2
            except Exception:
                lease_security_score = 0.5

        purpose_score = 0.5
        if requested_purpose:
            purpose_len = len(requested_purpose)
            if purpose_len >= 80:
                purpose_score = 1.0
            elif purpose_len >= 30:
                purpose_score = 0.75
            else:
                purpose_score = 0.35

        rules = {
            "rule_min_heads": r1,
            "rule_norm_positive": r2,
            "rule_amount_positive": r3,
            "rule_not_extreme_amount": r4,
            "rule_norm_in_range": r5,
            "profile_experience_score": round(float(exp_score), 4),
            "profile_capacity_score": round(float(capacity_score), 4),
            "profile_land_score": round(float(land_score), 4),
            "profile_revenue_score": round(float(revenue_score), 4),
            "profile_workforce_score": round(float(workforce_score), 4),
            "profile_infrastructure_score": round(float(infrastructure_score), 4),
            "profile_financial_discipline_score": round(float(financial_discipline_score), 4),
            "application_cofinancing_score": round(float(cofinancing_score), 4),
            "application_documentation_score": round(float(documentation_score), 4),
            "application_subsidy_history_score": round(float(subsidy_history_score), 4),
            "application_execution_score": round(float(execution_score), 4),
            "profile_lease_security_score": round(float(lease_security_score), 4),
            "application_purpose_score": round(float(purpose_score), 4),
        }
        base_rule_values = [r1, r2, r3, r4, r5]
        profile_values = [
            exp_score,
            capacity_score,
            land_score,
            revenue_score,
            workforce_score,
            infrastructure_score,
            financial_discipline_score,
            cofinancing_score,
            documentation_score,
            subsidy_history_score,
            execution_score,
            lease_security_score,
            purpose_score,
        ]
        rules["rule_score"] = float(np.mean(base_rule_values + profile_values))
        return rules

    def _build_criteria(self, data: dict, ml_score: float, rules: dict, anomaly: dict, composite: float) -> list:
        amount = float(data.get("amount", 0))
        norm = float(data.get("norm", 1))
        requested_heads = amount / norm if norm > 0 else 0
        profile_heads = data.get("profile_heads_count")
        profile_years = data.get("profile_years_active")
        profile_land = data.get("profile_land_area_ha")
        profile_revenue = data.get("profile_annual_revenue")
        profile_employees = data.get("profile_employee_count")
        own_contribution_percent = data.get("own_contribution_percent")
        planned_execution_months = data.get("planned_execution_months")
        has_supporting_documents = data.get("has_supporting_documents")
        previous_subsidies_count = data.get("previous_subsidies_count")
        previous_returns_count = data.get("previous_subsidy_returns_count")
        previous_violations_count = data.get("previous_subsidy_violations_count")
        previous_overdue_reports_count = data.get("previous_subsidy_overdue_reports_count")
        previous_closed_properly = data.get("previous_subsidies_closed_properly")
        requested_purpose = str(data.get("requested_purpose", "") or "").strip()
        lease_years_left = data.get("profile_lease_term_remaining_years")

        infrastructure_bits = {
            "livestock_facility": data.get("profile_has_livestock_facility"),
            "feed_base": data.get("profile_has_feed_base"),
            "equipment": data.get("profile_has_equipment"),
            "sales_contracts": data.get("profile_has_sales_contracts"),
        }

        criteria = [
            {
                "key": "ml_probability",
                "title": "ML вероятность одобрения",
                "score_0_1": round(float(ml_score), 4),
                "weight": 0.5,
                "status": "ok" if ml_score >= 0.5 else "risk",
                "explanation": "Оценка модели XGBoost на исторических заявках.",
                "source_ids": ["hist_features"],
            },
            {
                "key": "rule_block",
                "title": "Блок фиксированных правил",
                "score_0_1": round(float(rules.get("rule_score", 0)), 4),
                "weight": 0.3,
                "status": "ok" if rules.get("rule_score", 0) >= 0.6 else "risk",
                "explanation": "Проверки суммы, норматива, экстремальности и профильных ограничений.",
                "source_ids": ["hist_features", "constitution_kz", "real_subsidy_data_2025"],
            },
            {
                "key": "anomaly_block",
                "title": "Проверка на аномальность",
                "score_0_1": float(anomaly.get("anomaly_clean", 1)),
                "weight": 0.2,
                "status": "ok" if anomaly.get("anomaly_clean", 1) == 1 else "risk",
                "explanation": "Isolation Forest проверяет нетипичность заявки относительно исторического профиля.",
                "source_ids": ["hist_features"],
            },
        ]

        if profile_years is not None:
            criteria.append({
                "key": "profile_years_active",
                "title": "Стаж работы",
                "score_0_1": round(float(rules.get("profile_experience_score", 0.5)), 4),
                "status": "ok" if rules.get("profile_experience_score", 0.5) >= 0.5 else "risk",
                "value": profile_years,
                "explanation": "Низкий стаж снижает устойчивость заявки; высокий стаж повышает доверие.",
                "source_ids": ["constitution_kz"],
            })

        if profile_heads is not None:
            criteria.append({
                "key": "profile_heads_vs_request",
                "title": "Поголовье по профилю",
                "score_0_1": round(float(rules.get("profile_capacity_score", 0.5)), 4),
                "status": "ok" if rules.get("profile_capacity_score", 0.5) >= 0.7 else "risk",
                "value": {
                    "profile_heads_count": profile_heads,
                    "requested_heads_estimated": round(float(requested_heads), 2),
                },
                "explanation": "Если запрашиваемое поголовье сильно выше заявленного, score снижается.",
                "source_ids": ["hist_features"],
            })

        if profile_land is not None:
            criteria.append({
                "key": "profile_land_load",
                "title": "Нагрузка на гектар",
                "score_0_1": round(float(rules.get("profile_land_score", 0.5)), 4),
                "status": "ok" if rules.get("profile_land_score", 0.5) >= 0.6 else "risk",
                "value": {
                    "land_area_ha": profile_land,
                    "requested_heads_estimated": round(float(requested_heads), 2),
                },
                "explanation": "Слишком высокая плотность голов на гектар считается риском.",
                "source_ids": ["stat_livestock", "kazhydromet_meteo"],
            })

        if profile_revenue is not None:
            criteria.append({
                "key": "profile_revenue_coverage",
                "title": "Выручка хозяйства vs сумма заявки",
                "score_0_1": round(float(rules.get("profile_revenue_score", 0.5)), 4),
                "status": "ok" if rules.get("profile_revenue_score", 0.5) >= 0.6 else "risk",
                "value": {
                    "annual_revenue": round(float(profile_revenue), 2),
                    "requested_amount": round(float(amount), 2),
                },
                "explanation": "Если сумма заявки существенно выше годовой выручки, финансовый риск повышается.",
                "source_ids": ["hist_features"],
            })

        if profile_employees is not None:
            criteria.append({
                "key": "profile_workforce_capacity",
                "title": "Кадровая способность исполнить проект",
                "score_0_1": round(float(rules.get("profile_workforce_score", 0.5)), 4),
                "status": "ok" if rules.get("profile_workforce_score", 0.5) >= 0.6 else "risk",
                "value": {
                    "employee_count": profile_employees,
                    "requested_heads_estimated": round(float(requested_heads), 2),
                },
                "explanation": "Большой объём заявки при очень малом штате повышает операционный риск.",
                "source_ids": ["hist_features"],
            })

        if any(value is not None for value in infrastructure_bits.values()):
            ready_items = [key for key, value in infrastructure_bits.items() if value]
            criteria.append({
                "key": "profile_infrastructure_readiness",
                "title": "Инфраструктурная готовность",
                "score_0_1": round(float(rules.get("profile_infrastructure_score", 0.5)), 4),
                "status": "ok" if rules.get("profile_infrastructure_score", 0.5) >= 0.6 else "risk",
                "value": {
                    "ready_items": ready_items,
                    "declared_items_count": len([value for value in infrastructure_bits.values() if value is not None]),
                },
                "explanation": "Наличие помещений, кормовой базы, техники и сбыта повышает реалистичность исполнения.",
                "source_ids": ["farmer_registry_family_farms"],
            })

        if data.get("profile_has_tax_debt") is not None or data.get("profile_has_overdue_loans") is not None:
            criteria.append({
                "key": "profile_financial_discipline",
                "title": "Финансовая дисциплина хозяйства",
                "score_0_1": round(float(rules.get("profile_financial_discipline_score", 0.5)), 4),
                "status": "ok" if rules.get("profile_financial_discipline_score", 0.5) >= 0.6 else "risk",
                "value": {
                    "has_tax_debt": data.get("profile_has_tax_debt"),
                    "has_overdue_loans": data.get("profile_has_overdue_loans"),
                },
                "explanation": "Декларируемые налоговые долги и просроченные обязательства снижают доверие к исполнению.",
                "source_ids": [],
            })

        if lease_years_left is not None:
            criteria.append({
                "key": "profile_lease_security",
                "title": "Стабильность права на землю",
                "score_0_1": round(float(rules.get("profile_lease_security_score", 0.5)), 4),
                "status": "ok" if rules.get("profile_lease_security_score", 0.5) >= 0.5 else "risk",
                "value": {"lease_term_remaining_years": lease_years_left},
                "explanation": "Короткий остаток аренды повышает риск, что хозяйство не удержит ресурсную базу проекта.",
                "source_ids": [],
            })

        if own_contribution_percent is not None:
            criteria.append({
                "key": "application_cofinancing",
                "title": "Собственное софинансирование",
                "score_0_1": round(float(rules.get("application_cofinancing_score", 0.5)), 4),
                "status": "ok" if rules.get("application_cofinancing_score", 0.5) >= 0.5 else "risk",
                "value": {"own_contribution_percent": round(float(own_contribution_percent), 2)},
                "explanation": "Чем выше собственное участие, тем выше уверенность в заинтересованности и устойчивости проекта.",
                "source_ids": [],
            })

        if has_supporting_documents is not None:
            criteria.append({
                "key": "application_documentation",
                "title": "Подтверждающие документы",
                "score_0_1": round(float(rules.get("application_documentation_score", 0.5)), 4),
                "status": "ok" if rules.get("application_documentation_score", 0.5) >= 0.6 else "risk",
                "value": {"has_supporting_documents": has_supporting_documents},
                "explanation": "Заявки с договорами, счетами или коммерческими предложениями выглядят более подготовленными.",
                "source_ids": [],
            })

        history_values = [
            previous_subsidies_count,
            previous_returns_count,
            previous_violations_count,
            previous_overdue_reports_count,
            previous_closed_properly,
        ]
        if any(value is not None for value in history_values):
            criteria.append({
                "key": "application_subsidy_history",
                "title": "История исполнения прошлых субсидий",
                "score_0_1": round(float(rules.get("application_subsidy_history_score", 0.5)), 4),
                "status": "ok" if rules.get("application_subsidy_history_score", 0.5) >= 0.6 else "risk",
                "value": {
                    "previous_subsidies_count": previous_subsidies_count,
                    "returns_count": previous_returns_count,
                    "violations_count": previous_violations_count,
                    "overdue_reports_count": previous_overdue_reports_count,
                    "closed_properly": previous_closed_properly,
                },
                "explanation": "Возвраты, нарушения и просрочки отчётности резко снижают доверие к новой заявке.",
                "source_ids": [],
            })

        if planned_execution_months is not None:
            criteria.append({
                "key": "application_execution_horizon",
                "title": "Срок исполнения проекта",
                "score_0_1": round(float(rules.get("application_execution_score", 0.5)), 4),
                "status": "ok" if rules.get("application_execution_score", 0.5) >= 0.6 else "risk",
                "value": {"planned_execution_months": planned_execution_months},
                "explanation": "Слишком длинный горизонт исполнения повышает риск срыва и слабого контроля результата.",
                "source_ids": [],
            })

        if requested_purpose:
            criteria.append({
                "key": "application_purpose_clarity",
                "title": "Конкретность цели заявки",
                "score_0_1": round(float(rules.get("application_purpose_score", 0.5)), 4),
                "status": "ok" if rules.get("application_purpose_score", 0.5) >= 0.6 else "risk",
                "value": {"purpose_length": len(requested_purpose)},
                "explanation": "Чем конкретнее описана цель использования средств, тем проще проверить реализуемость проекта.",
                "source_ids": [],
            })

        # Add norm validation based on real subsidy data
        direction = data.get("direction", "")
        if direction:
            try:
                norm_validation = dataset_service.validate_norm_reasonableness(direction, norm)
                if norm_validation:
                    norm_status = "ok" if norm_validation.get("valid", True) else "risk"
                    criteria.append({
                        "key": "norm_reasonableness",
                        "title": "Реалистичность норматива",
                        "score_0_1": 1.0 if norm_status == "ok" else 0.5,
                        "status": norm_status,
                        "value": {
                            "norm": round(float(norm), 2),
                            "direction": direction,
                            "percentile": norm_validation.get("percentile"),
                        },
                        "explanation": norm_validation.get("reason", "Норматив в пределах исторических данных"),
                        "source_ids": ["real_subsidy_data_2025", "stat_livestock"],
                    })
            except Exception:
                pass

        # Add explicit point contribution
        for c in criteria:
            score_01 = c.get("score_0_1")
            weight = c.get("weight")

            if score_01 is not None:
                if weight:
                    max_points = round(float(weight) * 100.0, 2)
                    points = round(float(score_01) * max_points, 2)
                    c["max_points_0_100"] = max_points
                    c["points_0_100"] = points
                    c["points_formula"] = f"{float(score_01):.4f} * {max_points:.2f}"
                else:
                    c["max_points_0_100"] = "N/A"
                    c["points_0_100"] = "N/A"
                    c["points_formula"] = "Справочно"

        return criteria

    def _build_human_explanation(self, criteria: list, tier: str, composite: float) -> str:
        risky = [c for c in criteria if c.get("status") == "risk"]
        if not risky:
            return f"Заявка получила {composite:.1f} балла ({tier}). Критических рисков по правилам и профилю не выявлено."
        top = ", ".join(c.get("title", c.get("key", "критерий")) for c in risky[:3])
        return (
            f"Заявка получила {composite:.1f} балла ({tier}). "
            f"Ключевые факторы снижения: {top}."
        )

    def _build_full_context_template_explanation(self, data: dict, criteria: list, tier: str, composite: float) -> str:
        amount = float(data.get("amount", 0))
        norm = float(data.get("norm", 0))
        direction = str(data.get("direction", "") or "не указано")
        region = str(data.get("region", "") or "не указан")
        requested_heads = amount / norm if norm > 0 else 0
        risky = [c for c in criteria if c.get("status") == "risk"]
        ok_items = [c for c in criteria if c.get("status") == "ok"]
        risky_titles = ", ".join(c.get("title", "критерий") for c in risky[:4]) or "критических рисков не выявлено"
        strong_titles = ", ".join(c.get("title", "критерий") for c in ok_items[:4]) or "выраженных сильных сторон не выделено"
        return (
            f"Заявка по направлению {direction} из региона {region} получила {composite:.1f} балла и отнесена к категории '{tier}'. "
            f"Запрошенная сумма составляет {amount:,.0f} тг при нормативе {norm:,.0f} тг, что соответствует расчетному объему {requested_heads:.2f}. "
            f"Сильные стороны заявки: {strong_titles}. "
            f"Основные риски: {risky_titles}. "
            f"Полная оценка строится на ML-модели, rule-based критериях, данных профиля хозяйства и заявочных параметрах, поэтому итог нужно читать в связке со всеми критериями ниже."
        )

    def _resolve_sources(self, source_ids: list) -> list:
        if not source_ids:
            source_ids = []
        selected = []
        seen = set()
        lookup = {str(s.get("id")): s for s in self.sources if isinstance(s, dict)}
        for sid in source_ids:
            src = lookup.get(str(sid))
            if src and str(sid) not in seen:
                selected.append(src)
                seen.add(str(sid))
        # Always include legal and historical references in the response baseline
        for sid in ("hist_features", "constitution_kz"):
            src = lookup.get(sid)
            if src and sid not in seen:
                selected.append(src)
                seen.add(sid)
        return selected

    def check_anomaly(self, features: np.ndarray) -> dict:
        if self.iso_forest is None:
            return {"anomaly_clean": 1, "anomaly_score_norm": 1.0}
        anom_feat_idx = [FEATURE_COLS.index(f) for f in ANOMALY_FEATURES]
        X_a = features[0, anom_feat_idx].reshape(1, -1)
        label = self.iso_forest.predict(X_a)[0]
        raw = self.iso_forest.decision_function(X_a)[0]
        norm_score = max(0, min(1, (raw + 0.5)))
        return {"anomaly_clean": int(label == 1), "anomaly_score_norm": round(float(norm_score), 4)}

    def compute_shap(self, features: np.ndarray) -> list:
        X_df = pd.DataFrame(features, columns=FEATURE_COLS)
        sv = self.explainer(X_df)
        vals = sv.values[0]
        pairs = list(zip(FEATURE_COLS, vals))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top5 = []
        for feat, val in pairs[:5]:
            raw_value = float(features[0, FEATURE_COLS.index(feat)])
            top5.append({
                "feature": feat,
                "label": FEATURE_LABELS_RU.get(feat, feat),
                "shap_value": round(float(val), 4),
                "raw_value": round(raw_value, 4),
                "direction": "positive" if val > 0 else "negative",
            })
        return top5

    def score(self, data: dict) -> dict:
        self.load()
        features = self.build_features(data)
        ml_score = float(self.model.predict_proba(features)[0, 1])
        rules = self.check_rules(data)
        anomaly = self.check_anomaly(features)
        shap_explanations = self.compute_shap(features)

        composite = (ml_score * 0.5 + rules["rule_score"] * 0.3 + anomaly["anomaly_clean"] * 0.2) * 100
        composite = round(composite, 2)

        if composite >= 75:
            tier = "Приоритет"
        elif composite >= 50:
            tier = "Рассмотреть"
        else:
            tier = "Отклонить"

        criteria = self._build_criteria(data, ml_score, rules, anomaly, composite)
        source_ids = []
        for c in criteria:
            source_ids.extend(c.get("source_ids", []))
        resolved_sources = self._resolve_sources(source_ids)
        template_explanation = self._build_human_explanation(criteria, tier, composite)

        # Try LLM explanation — falls back to template if key not set or Groq unavailable
        from llm_explain import generate_llm_explanation
        llm_result = generate_llm_explanation(
            composite=composite,
            tier=tier,
            criteria=criteria,
            amount=float(data.get("amount", 0)),
            direction=str(data.get("direction", "")),
            region=str(data.get("region", "")),
            profile_years=data.get("profile_years_active"),
            profile_heads=data.get("profile_heads_count"),
            profile_land=data.get("profile_land_area_ha"),
            ml_score=ml_score,
            rule_score=rules.get("rule_score"),
            rules_detail=rules,
            extra_context=str(data.get("extra_context", "") or ""),
        )
        explanation = llm_result["explanation"] or template_explanation
        full_context_explanation = llm_result["full_context_explanation"] or self._build_full_context_template_explanation(
            data, criteria, tier, composite
        )
        explanation_source = llm_result["explanation_source"]

        return {
            "ml_score": round(ml_score, 4),
            "rule_score": round(rules["rule_score"], 4),
            "anomaly_clean": anomaly["anomaly_clean"],
            "composite_score": composite,
            "tier": tier,
            "rules": rules,
            "shap_explanations": shap_explanations,
            "criteria": criteria,
            "explanation": explanation,
            "full_context_explanation": full_context_explanation,
            "explanation_source": explanation_source,
            "llm_extra_comment": llm_result.get("extra_comment") or "",
            "sources": resolved_sources,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }


# Singleton
scoring_service = ScoringService()
