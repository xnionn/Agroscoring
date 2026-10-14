from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
import json
from database import get_db, User, FarmerProfile, Application
from auth import get_current_user, require_commissioner
from scoring import scoring_service
from report_service import generate_html_report, generate_json_report
from email_service import send_email_html

router = APIRouter(prefix="/api/applications", tags=["applications"])

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


def expand_region_filter(region: str) -> list[str]:
    for variants in REGION_ALIASES.values():
        if region in variants:
            return variants
    return [region]


from pydantic import BaseModel, Field

class ApplicationCreate(BaseModel):
    direction: str
    subsidy_name: str
    amount: float = Field(..., gt=0, description="Сумма субсидии должна быть положительной")
    norm: float = Field(..., gt=0, description="Норматив должен быть положительным")
    region: Optional[str] = None
    district: Optional[str] = None
    own_contribution_percent: Optional[float] = Field(None, ge=0, le=100)
    planned_execution_months: Optional[int] = Field(None, ge=1, le=60)
    has_supporting_documents: Optional[bool] = None
    previous_subsidies_count: Optional[int] = Field(None, ge=0)
    previous_subsidy_returns_count: Optional[int] = Field(None, ge=0)
    previous_subsidy_violations_count: Optional[int] = Field(None, ge=0)
    previous_subsidy_overdue_reports_count: Optional[int] = Field(None, ge=0)
    previous_subsidies_closed_properly: Optional[bool] = None
    requested_purpose: Optional[str] = Field(None, max_length=2000)
    extra_context: Optional[str] = Field(None, max_length=3000)


class DecisionUpdate(BaseModel):
    decision: str  # approved / rejected / more_info
    note: Optional[str] = None


@router.post("")
def submit_application(req: ApplicationCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(FarmerProfile).filter(FarmerProfile.user_id == current_user.id).first()
    region = req.region or (profile.region if profile else "")
    district = req.district or (profile.district if profile else "")

    if not region:
        raise HTTPException(400, "Region is required. Fill your profile or specify in the request")
    if req.amount <= 0:
        raise HTTPException(400, "Сумма субсидии (amount) должна быть больше 0")
    if req.norm <= 0:
        raise HTTPException(400, "Норматив (norm) должен быть больше 0")

    # Score the application
    score_data = {
        "amount": req.amount, "norm": req.norm,
        "region": region, "district": district,
        "direction": req.direction, "subsidy_name": req.subsidy_name,
        "profile_heads_count": profile.heads_count if profile else None,
        "profile_years_active": profile.years_active if profile else None,
        "profile_land_area_ha": profile.land_area_ha if profile else None,
        "profile_annual_revenue": profile.annual_revenue if profile else None,
        "profile_employee_count": profile.employee_count if profile else None,
        "profile_owned_land_area_ha": profile.owned_land_area_ha if profile else None,
        "profile_leased_land_area_ha": profile.leased_land_area_ha if profile else None,
        "profile_lease_term_remaining_years": profile.lease_term_remaining_years if profile else None,
        "profile_has_livestock_facility": profile.has_livestock_facility if profile else None,
        "profile_has_feed_base": profile.has_feed_base if profile else None,
        "profile_has_equipment": profile.has_equipment if profile else None,
        "profile_has_sales_contracts": profile.has_sales_contracts if profile else None,
        "profile_has_tax_debt": profile.has_tax_debt if profile else None,
        "profile_has_overdue_loans": profile.has_overdue_loans if profile else None,
        "own_contribution_percent": req.own_contribution_percent,
        "planned_execution_months": req.planned_execution_months,
        "has_supporting_documents": req.has_supporting_documents,
        "previous_subsidies_count": req.previous_subsidies_count,
        "previous_subsidy_returns_count": req.previous_subsidy_returns_count,
        "previous_subsidy_violations_count": req.previous_subsidy_violations_count,
        "previous_subsidy_overdue_reports_count": req.previous_subsidy_overdue_reports_count,
        "previous_subsidies_closed_properly": req.previous_subsidies_closed_properly,
        "requested_purpose": req.requested_purpose,
        "extra_context": req.extra_context,
    }
    result = scoring_service.score(score_data)
    rules = result["rules"]
    heads = req.amount / req.norm if req.norm > 0 else 0
    explain_payload = {
        "shap_explanations": result.get("shap_explanations", []),
        "criteria": result.get("criteria", []),
        "explanation": result.get("explanation", ""),
        "full_context_explanation": result.get("full_context_explanation", ""),
        "explanation_source": result.get("explanation_source", "template"),
        "llm_extra_comment": result.get("llm_extra_comment", ""),
        "sources": result.get("sources", []),
        "calculated_at": result.get("calculated_at"),
    }

    app = Application(
        user_id=current_user.id,
        farmer_profile_id=profile.id if profile else None,
        farmer_name=current_user.full_name,
        direction=req.direction,
        subsidy_name=req.subsidy_name,
        subsidy_category=scoring_service.classify_subsidy(req.subsidy_name),
        amount=req.amount, norm=req.norm, heads=round(heads, 2),
        region=region, district=district,
        own_contribution_percent=req.own_contribution_percent,
        planned_execution_months=req.planned_execution_months,
        has_supporting_documents=req.has_supporting_documents,
        previous_subsidies_count=req.previous_subsidies_count,
        previous_subsidy_returns_count=req.previous_subsidy_returns_count,
        previous_subsidy_violations_count=req.previous_subsidy_violations_count,
        previous_subsidy_overdue_reports_count=req.previous_subsidy_overdue_reports_count,
        previous_subsidies_closed_properly=req.previous_subsidies_closed_properly,
        requested_purpose=req.requested_purpose,
        extra_context=req.extra_context,
        llm_extra_comment=result.get("llm_extra_comment"),
        ml_score=result["ml_score"],
        rule_score=result["rule_score"],
        anomaly_clean=result["anomaly_clean"],
        composite_score=result["composite_score"],
        tier=result["tier"],
        rule_min_heads=rules.get("rule_min_heads"),
        rule_norm_positive=rules.get("rule_norm_positive"),
        rule_amount_positive=rules.get("rule_amount_positive"),
        rule_not_extreme_amount=rules.get("rule_not_extreme_amount"),
        rule_norm_in_range=rules.get("rule_norm_in_range"),
        shap_json=json.dumps(explain_payload, ensure_ascii=False),
        decision="pending",
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return _app_to_dict(app)


@router.get("")
def list_applications(
    region: Optional[str] = None,
    direction: Optional[str] = None,
    tier: Optional[str] = None,
    min_score: Optional[float] = None,
    decision: Optional[str] = None,
    limit: int = Query(200, le=500),
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Application)
    if current_user.role == "farmer":
        q = q.filter(Application.user_id == current_user.id)

    if region:
        region_variants = expand_region_filter(region)
        q = q.filter(Application.region.in_(region_variants))

    if direction:
        q = q.filter(Application.direction == direction)
    if tier:
        q = q.filter(Application.tier == tier)
    if decision:
        q = q.filter(Application.decision == decision)
    if min_score is not None:
        q = q.filter(Application.composite_score >= min_score)

    total = q.count()
    apps = q.order_by(Application.composite_score.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": [_app_to_dict(a) for a in apps]}


@router.get("/{app_id}")
def get_application(app_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(404, "Application not found")
    if current_user.role == "farmer" and app.user_id != current_user.id:
        raise HTTPException(403, "Access denied")
    _backfill_shap_if_missing(app, db)
    return _app_to_dict(app)


@router.put("/{app_id}/decision")
def make_decision(app_id: int, req: DecisionUpdate, current_user: User = Depends(require_commissioner), db: Session = Depends(get_db)):
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(404, "Application not found")
    if req.decision not in ("approved", "rejected", "more_info"):
        raise HTTPException(400, "Decision must be approved, rejected, or more_info")
    app.decision = req.decision
    app.decision_note = req.note
    app.decision_by = current_user.full_name
    app.decision_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(app)
    return _app_to_dict(app)


def _backfill_shap_if_missing(app: Application, db: Session) -> None:
    if app.shap_json and str(app.shap_json).strip():
        try:
            parsed = json.loads(app.shap_json)
            if isinstance(parsed, dict) and parsed.get("full_context_explanation"):
                return
        except Exception:
            pass
    try:
        score_input = {
            "amount": float(app.amount or 0),
            "norm": float(app.norm or 0),
            "region": app.region or "",
            "district": app.district or "",
            "direction": app.direction or "",
            "subsidy_name": app.subsidy_name or "",
            "own_contribution_percent": app.own_contribution_percent,
            "planned_execution_months": app.planned_execution_months,
            "has_supporting_documents": app.has_supporting_documents,
            "previous_subsidies_count": app.previous_subsidies_count,
            "previous_subsidy_returns_count": app.previous_subsidy_returns_count,
            "previous_subsidy_violations_count": app.previous_subsidy_violations_count,
            "previous_subsidy_overdue_reports_count": app.previous_subsidy_overdue_reports_count,
            "previous_subsidies_closed_properly": app.previous_subsidies_closed_properly,
            "requested_purpose": app.requested_purpose,
            "extra_context": app.extra_context,
        }

        if app.farmer_profile_id:
            profile = db.query(FarmerProfile).filter(FarmerProfile.id == app.farmer_profile_id).first()
            if profile:
                score_input.update({
                    "profile_heads_count": profile.heads_count,
                    "profile_years_active": profile.years_active,
                    "profile_land_area_ha": profile.land_area_ha,
                    "profile_annual_revenue": profile.annual_revenue,
                    "profile_employee_count": profile.employee_count,
                    "profile_owned_land_area_ha": profile.owned_land_area_ha,
                    "profile_leased_land_area_ha": profile.leased_land_area_ha,
                    "profile_lease_term_remaining_years": profile.lease_term_remaining_years,
                    "profile_has_livestock_facility": profile.has_livestock_facility,
                    "profile_has_feed_base": profile.has_feed_base,
                    "profile_has_equipment": profile.has_equipment,
                    "profile_has_sales_contracts": profile.has_sales_contracts,
                    "profile_has_tax_debt": profile.has_tax_debt,
                    "profile_has_overdue_loans": profile.has_overdue_loans,
                })

        result = scoring_service.score(score_input)
        explain_payload = {
            "shap_explanations": result.get("shap_explanations", []),
            "criteria": result.get("criteria", []),
            "explanation": result.get("explanation", ""),
            "full_context_explanation": result.get("full_context_explanation", ""),
            "explanation_source": result.get("explanation_source", "template"),
            "llm_extra_comment": result.get("llm_extra_comment", ""),
            "sources": result.get("sources", []),
            "calculated_at": result.get("calculated_at"),
        }
        if explain_payload.get("shap_explanations") or explain_payload.get("criteria"):
            app.shap_json = json.dumps(explain_payload, ensure_ascii=False)
            app.llm_extra_comment = result.get("llm_extra_comment")
            db.commit()
            db.refresh(app)
    except Exception:
        # SHAP is optional for historical rows; keep response resilient.
        pass


def _app_to_dict(a: Application) -> dict:
    shap_data = []
    criteria_data = []
    explanation_text = ""
    full_context_explanation_text = ""
    explanation_source = "template"
    llm_extra_comment = a.llm_extra_comment or ""
    sources_data = []
    calculated_at = None
    if a.shap_json:
        try:
            parsed = json.loads(a.shap_json)
            if isinstance(parsed, list):
                # Legacy format: only SHAP array
                shap_data = parsed
            elif isinstance(parsed, dict):
                shap_data = parsed.get("shap_explanations", []) or []
                criteria_data = parsed.get("criteria", []) or []
                explanation_text = parsed.get("explanation", "") or ""
                full_context_explanation_text = parsed.get("full_context_explanation", "") or ""
                explanation_source = parsed.get("explanation_source", "template") or "template"
                llm_extra_comment = parsed.get("llm_extra_comment", "") or llm_extra_comment
                sources_data = parsed.get("sources", []) or []
                calculated_at = parsed.get("calculated_at")
        except Exception:
            pass
    return {
        "id": a.id,
        "farmer_name": a.farmer_name,
        "direction": a.direction,
        "subsidy_name": a.subsidy_name,
        "subsidy_category": a.subsidy_category,
        "amount": a.amount, "norm": a.norm, "heads": a.heads,
        "region": a.region, "district": a.district,
        "own_contribution_percent": a.own_contribution_percent,
        "planned_execution_months": a.planned_execution_months,
        "has_supporting_documents": a.has_supporting_documents,
        "previous_subsidies_count": a.previous_subsidies_count,
        "previous_subsidy_returns_count": a.previous_subsidy_returns_count,
        "previous_subsidy_violations_count": a.previous_subsidy_violations_count,
        "previous_subsidy_overdue_reports_count": a.previous_subsidy_overdue_reports_count,
        "previous_subsidies_closed_properly": a.previous_subsidies_closed_properly,
        "requested_purpose": a.requested_purpose,
        "extra_context": a.extra_context,
        "date_submitted": a.date_submitted.isoformat() if a.date_submitted else None,
        "ml_score": a.ml_score,
        "rule_score": a.rule_score,
        "anomaly_clean": a.anomaly_clean,
        "composite_score": a.composite_score,
        "tier": a.tier,
        "rules": {
            "rule_min_heads": a.rule_min_heads,
            "rule_norm_positive": a.rule_norm_positive,
            "rule_amount_positive": a.rule_amount_positive,
            "rule_not_extreme_amount": a.rule_not_extreme_amount,
            "rule_norm_in_range": a.rule_norm_in_range,
        },
        "shap_explanations": shap_data,
        "criteria": criteria_data,
        "explanation": explanation_text,
        "full_context_explanation": full_context_explanation_text,
        "explanation_source": explanation_source,
        "llm_extra_comment": llm_extra_comment,
        "sources": sources_data,
        "calculated_at": calculated_at,
        "decision": a.decision,
        "decision_note": a.decision_note,
        "decision_by": a.decision_by,
        "decision_at": a.decision_at.isoformat() if a.decision_at else None,
        "is_demo": a.is_demo,
    }


@router.get("/{app_id}/export/html")
def export_application_html(app_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Export application report as HTML (viewable in browser, printable to PDF)."""
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(404, "Application not found")
    if current_user.role == "farmer" and app.user_id != current_user.id:
        raise HTTPException(403, "Access denied")
    
    _backfill_shap_if_missing(app, db)
    app_dict = _app_to_dict(app)
    html_content = generate_html_report(app_dict)
    
    return HTMLResponse(
        content=html_content,
        headers={
            "Content-Disposition": f"inline; filename='agrosubsidia_app_{app_id}.html'"
        }
    )


@router.get("/{app_id}/export/json")
def export_application_json(app_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Export application report as machine-readable JSON."""
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(404, "Application not found")
    if current_user.role == "farmer" and app.user_id != current_user.id:
        raise HTTPException(403, "Access denied")
    
    _backfill_shap_if_missing(app, db)
    app_dict = _app_to_dict(app)
    json_content = generate_json_report(app_dict)
    
    return {
        "status": "ok",
        "format": "json",
        "data": json.loads(json_content)
    }


@router.post("/{app_id}/export/send-to-email")
def send_report_to_email(
    app_id: int,
    req: Optional[dict] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send HTML report to user's email address."""
    app = db.query(Application).filter(Application.id == app_id).first()
    if not app:
        raise HTTPException(404, "Application not found")
    if current_user.role == "farmer" and app.user_id != current_user.id:
        raise HTTPException(403, "Access denied")
    
    # Backfill missing dependencies
    _backfill_shap_if_missing(app, db)
    app_dict = _app_to_dict(app)
    html_content = generate_html_report(app_dict)
    
    # Send email
    subject = f"Отчет по заявке #{app_id} - {app.subsidy_name}"
    success, message = send_email_html(current_user.email, subject, html_content)
    
    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {
        "status": "success",
        "message": message,
        "report_size_kb": len(html_content) / 1024,
    }
