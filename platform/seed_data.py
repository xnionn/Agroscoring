"""
Seed demo data: loads top records from final_scored.pkl into the database
so the dashboard isn't empty on first launch.
"""
import os
import sys
import pickle
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal, Application, init_db


def seed_demo_data(max_rows=150):
    init_db()
    db = SessionLocal()

    pkl_path = os.path.join(BASE_DIR, "final_scored.pkl")
    if not os.path.exists(pkl_path):
        print("[Seed] final_scored.pkl not found, skipping")
        db.close()
        return

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    # Take top by composite_score + some mid + some low for diversity
    top = df.nlargest(80, "composite_score")
    mid = df[(df["composite_score"] >= 40) & (df["composite_score"] < 75)].head(50)
    low = df[df["composite_score"] < 40].head(20)
    sample = __import__("pandas").concat([top, mid, low]).drop_duplicates(subset=["num"])[:max_rows]

    existing_demo_names = {
        name for (name,) in db.query(Application.farmer_name)
        .filter(Application.is_demo == True)
        .all()
        if name
    }
    existing_demo_count = len(existing_demo_names)
    if existing_demo_count >= max_rows:
        print(f"[Seed] Demo rows already present: {existing_demo_count}, target: {max_rows}. Nothing to add.")
        db.close()
        return

    count = 0
    for _, row in sample.iterrows():
        demo_name = f"Фермер #{int(row.get('num', 0))}"
        if demo_name in existing_demo_names:
            continue
        if existing_demo_count + count >= max_rows:
            break

        heads_val = row.get("heads", 0)
        if hasattr(heads_val, 'item'):
            heads_val = heads_val.item()
        heads_val = float(heads_val) if not (isinstance(heads_val, float) and np.isnan(heads_val)) else 0

        app = Application(
            farmer_name=demo_name,
            direction=str(row.get("direction", "")),
            subsidy_name=str(row.get("subsidy_name", "")),
            subsidy_category=str(row.get("subsidy_category", "")) if "subsidy_category" in row.index else "",
            amount=float(row.get("amount", 0)),
            norm=float(row.get("norm", 0)),
            heads=round(heads_val, 2),
            region=str(row.get("region", "")),
            district=str(row.get("district", "")) if not (isinstance(row.get("district"), float) and np.isnan(row.get("district", 0))) else "",
            date_submitted=row.get("date"),
            is_demo=True,
            ml_score=round(float(row.get("ml_score", 0)), 4),
            rule_score=round(float(row.get("rule_score", 1)), 4),
            anomaly_clean=int(row.get("anomaly_clean", 1)),
            composite_score=round(float(row.get("composite_score", 0)), 2),
            tier=str(row.get("tier", "")).replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", ""),
            rule_min_heads=int(row.get("rule_min_heads", 1)) if "rule_min_heads" in row.index else 1,
            rule_norm_positive=int(row.get("rule_norm_positive", 1)) if "rule_norm_positive" in row.index else 1,
            rule_amount_positive=int(row.get("rule_amount_positive", 1)) if "rule_amount_positive" in row.index else 1,
            rule_not_extreme_amount=int(row.get("rule_not_extreme_amount", 1)) if "rule_not_extreme_amount" in row.index else 1,
            rule_norm_in_range=int(row.get("rule_norm_in_range", 1)) if "rule_norm_in_range" in row.index else 1,
            decision="pending",
        )
        db.add(app)
        count += 1

    db.commit()
    db.close()
    print(f"[Seed] Existing demo rows: {existing_demo_count}")
    print(f"[Seed] Inserted missing demo rows: {count}")
    print(f"[Seed] Final demo rows target: up to {max_rows}")


if __name__ == "__main__":
    seed_demo_data()
