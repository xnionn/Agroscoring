import json
import os
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import get_db, Application, User
from auth import get_current_user

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

REGION_ALIASES = {
    "Абай облысы": ["Абай облысы", "Абайская область"],
    "Ақмола облысы": ["Ақмола облысы", "Акмолинская область"],
    "Ақтөбе облысы": ["Ақтөбе облысы", "Актюбинская область"],
    "Алматы облысы": ["Алматы облысы", "Алматинская область"],
    "Атырау облысы": ["Атырау облысы", "Атырауская область"],
    "Батыс Қазақстан облысы": ["Батыс Қазақстан облысы", "Западно-Казахстанская область"],
    "Жамбыл облысы": ["Жамбыл облысы", "Жамбылская область"],
    "Жетісу облысы": ["Жетісу облысы", "Жетысуская область"],
    "Қарағанды облысы": ["Қарағанды облысы", "Карагандинская область"],
    "Қостанай облысы": ["Қостанай облысы", "Костанайская область"],
    "Қызылорда облысы": ["Қызылорда облысы", "Кызылординская область"],
    "Маңғыстау облысы": ["Маңғыстау облысы", "Мангистауская область"],
    "Павлодар облысы": ["Павлодар облысы", "Павлодарская область"],
    "Солтүстік Қазақстан облысы": ["Солтүстік Қазақстан облысы", "Северо-Казахстанская область"],
    "Түркістан облысы": ["Түркістан облысы", "Туркестанская область"],
    "Ұлытау облысы": ["Ұлытау облысы", "Улытауская область"],
    "Шығыс Қазақстан облысы": ["Шығыс Қазақстан облысы", "Восточно-Казахстанская область"],
}

REGIONS = [
    "Абай облысы", "Ақмола облысы", "Ақтөбе облысы", "Алматы облысы",
    "Атырау облысы", "Батыс Қазақстан облысы", "Жамбыл облысы",
    "Жетісу облысы", "Қарағанды облысы", "Қостанай облысы",
    "Қызылорда облысы", "Маңғыстау облысы", "Павлодар облысы",
    "Солтүстік Қазақстан облысы", "Түркістан облысы",
    "Ұлытау облысы", "Шығыс Қазақстан облысы",
]

MAPPING = {}
mapping_path = os.path.join(os.path.dirname(__file__), "..", "..", "mapping.json")
try:
    with open(mapping_path, "r", encoding="utf-8") as f:
        MAPPING = json.load(f)
except Exception:
    pass

DIRECTIONS = list(MAPPING.keys()) if MAPPING else []


def expand_region_filter(region: str) -> list[str]:
    for variants in REGION_ALIASES.values():
        if region in variants:
            return variants
    return [region]


@router.get("/summary")
def summary(
    region: Optional[str] = None,
    direction: Optional[str] = None,
    tier: Optional[str] = None,
    min_score: Optional[float] = None,
    decision: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Application)
    if current_user.role == "farmer":
        q = q.filter(Application.user_id == current_user.id)

    if region:
        q = q.filter(Application.region.in_(expand_region_filter(region)))
    if direction:
        q = q.filter(Application.direction == direction)
    if tier:
        q = q.filter(Application.tier == tier)
    if decision:
        q = q.filter(Application.decision == decision)
    if min_score is not None:
        q = q.filter(Application.composite_score >= min_score)

    total = q.count()
    priority = q.filter(Application.tier == "Приоритет").count()
    avg_score = q.with_entities(func.avg(Application.composite_score)).scalar() or 0
    total_amount = q.with_entities(func.sum(Application.amount)).scalar() or 0
    pending = q.filter(Application.decision == "pending").count()
    approved = q.filter(Application.decision == "approved").count()
    rejected = q.filter(Application.decision == "rejected").count()

    # Tier distribution
    tiers = {}
    for tier_name in ["Приоритет", "Рассмотреть", "Отклонить"]:
        cnt = q.filter(Application.tier == tier_name).count()
        tiers[tier_name] = cnt

    # Avg score by region
    region_scores = (
        q.with_entities(Application.region, func.avg(Application.composite_score), func.count(Application.id))
        .group_by(Application.region)
        .all()
    )
    regions_data = [{"region": r, "avg_score": round(s, 1), "count": c} for r, s, c in region_scores]

    return {
        "total": total, "priority": priority,
        "avg_score": round(avg_score, 1),
        "total_amount_bln": round(total_amount / 1_000_000_000, 2),
        "pending": pending, "approved": approved, "rejected": rejected,
        "tiers": tiers, "regions": regions_data,
    }


@router.get("/reference-data")
def reference_data():
    return {"regions": REGIONS, "directions": DIRECTIONS, "mapping": MAPPING}
