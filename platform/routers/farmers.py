from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database import get_db, User, FarmerProfile, Application
from auth import get_current_user

router = APIRouter(prefix="/api/farmers", tags=["farmers"])


class ProfileCreate(BaseModel):
    iin: Optional[str] = None
    phone: Optional[str] = None
    farm_name: Optional[str] = None
    region: str
    district: Optional[str] = None
    livestock_direction: Optional[str] = None
    heads_count: Optional[int] = None
    years_active: Optional[int] = None
    land_area_ha: Optional[float] = None
    annual_revenue: Optional[float] = None
    employee_count: Optional[int] = None
    owned_land_area_ha: Optional[float] = None
    leased_land_area_ha: Optional[float] = None
    lease_term_remaining_years: Optional[int] = None
    has_livestock_facility: Optional[bool] = None
    has_feed_base: Optional[bool] = None
    has_equipment: Optional[bool] = None
    has_sales_contracts: Optional[bool] = None
    has_tax_debt: Optional[bool] = None
    has_overdue_loans: Optional[bool] = None


@router.post("/profile")
def create_or_update_profile(req: ProfileCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(FarmerProfile).filter(FarmerProfile.user_id == current_user.id).first()
    if profile:
        for k, v in req.dict(exclude_unset=True).items():
            setattr(profile, k, v)
    else:
        profile = FarmerProfile(user_id=current_user.id, **req.dict())
        db.add(profile)
    db.commit()
    db.refresh(profile)
    return {
        "id": profile.id, "farm_name": profile.farm_name,
        "region": profile.region, "district": profile.district,
        "livestock_direction": profile.livestock_direction,
        "heads_count": profile.heads_count, "years_active": profile.years_active,
        "land_area_ha": profile.land_area_ha,
        "annual_revenue": profile.annual_revenue,
        "employee_count": profile.employee_count,
        "owned_land_area_ha": profile.owned_land_area_ha,
        "leased_land_area_ha": profile.leased_land_area_ha,
        "lease_term_remaining_years": profile.lease_term_remaining_years,
        "has_livestock_facility": profile.has_livestock_facility,
        "has_feed_base": profile.has_feed_base,
        "has_equipment": profile.has_equipment,
        "has_sales_contracts": profile.has_sales_contracts,
        "has_tax_debt": profile.has_tax_debt,
        "has_overdue_loans": profile.has_overdue_loans,
    }


@router.get("/profile")
def get_my_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(FarmerProfile).filter(FarmerProfile.user_id == current_user.id).first()
    if not profile:
        return None
    return {
        "id": profile.id, "farm_name": profile.farm_name,
        "region": profile.region, "district": profile.district,
        "livestock_direction": profile.livestock_direction,
        "heads_count": profile.heads_count, "years_active": profile.years_active,
        "land_area_ha": profile.land_area_ha, "iin": profile.iin, "phone": profile.phone,
        "annual_revenue": profile.annual_revenue,
        "employee_count": profile.employee_count,
        "owned_land_area_ha": profile.owned_land_area_ha,
        "leased_land_area_ha": profile.leased_land_area_ha,
        "lease_term_remaining_years": profile.lease_term_remaining_years,
        "has_livestock_facility": profile.has_livestock_facility,
        "has_feed_base": profile.has_feed_base,
        "has_equipment": profile.has_equipment,
        "has_sales_contracts": profile.has_sales_contracts,
        "has_tax_debt": profile.has_tax_debt,
        "has_overdue_loans": profile.has_overdue_loans,
    }

@router.get("/applications")
def get_my_applications(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(FarmerProfile).filter(FarmerProfile.user_id == current_user.id).first()
    if not profile:
        return []

    # Applications are linked through farmer_profile_id/user_id in current schema.
    apps = (
        db.query(Application)
        .filter(Application.farmer_profile_id == profile.id)
        .order_by(Application.id.desc())
        .all()
    )

    result = []
    for app in apps:
        result.append({
            "id": app.id,
            "direction": app.direction,
            "subsidy_name": app.subsidy_name,
            "amount": app.amount,
            "norm": app.norm,
            "composite_score": app.composite_score,
            "tier": app.tier,
            "decision": app.decision,
            "decision_note": app.decision_note,
            "date_submitted": app.date_submitted.isoformat() if app.date_submitted else None,
        })
    return result
