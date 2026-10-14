from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "agrosubsidia.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(String, default="farmer")  # "farmer" | "commissioner"
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    farmer_profile = relationship("FarmerProfile", back_populates="user", uselist=False)
    applications = relationship("Application", back_populates="user")


class FarmerProfile(Base):
    __tablename__ = "farmer_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    iin = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    farm_name = Column(String, nullable=True)
    region = Column(String, nullable=False)
    district = Column(String, nullable=True)
    livestock_direction = Column(String, nullable=True)
    heads_count = Column(Integer, nullable=True)
    years_active = Column(Integer, nullable=True)
    land_area_ha = Column(Float, nullable=True)
    annual_revenue = Column(Float, nullable=True)
    employee_count = Column(Integer, nullable=True)
    owned_land_area_ha = Column(Float, nullable=True)
    leased_land_area_ha = Column(Float, nullable=True)
    lease_term_remaining_years = Column(Integer, nullable=True)
    has_livestock_facility = Column(Boolean, nullable=True)
    has_feed_base = Column(Boolean, nullable=True)
    has_equipment = Column(Boolean, nullable=True)
    has_sales_contracts = Column(Boolean, nullable=True)
    has_tax_debt = Column(Boolean, nullable=True)
    has_overdue_loans = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="farmer_profile")
    applications = relationship("Application", back_populates="farmer_profile")


class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    farmer_profile_id = Column(Integer, ForeignKey("farmer_profiles.id"), nullable=True)
    farmer_name = Column(String, nullable=True)

    # Application fields
    direction = Column(String, nullable=False)
    subsidy_name = Column(String, nullable=False)
    subsidy_category = Column(String, nullable=True)
    amount = Column(Float, nullable=False)
    norm = Column(Float, nullable=False)
    heads = Column(Float, nullable=True)
    region = Column(String, nullable=False)
    district = Column(String, nullable=True)
    own_contribution_percent = Column(Float, nullable=True)
    planned_execution_months = Column(Integer, nullable=True)
    has_supporting_documents = Column(Boolean, nullable=True)
    previous_subsidies_count = Column(Integer, nullable=True)
    previous_subsidy_returns_count = Column(Integer, nullable=True)
    previous_subsidy_violations_count = Column(Integer, nullable=True)
    previous_subsidy_overdue_reports_count = Column(Integer, nullable=True)
    previous_subsidies_closed_properly = Column(Boolean, nullable=True)
    requested_purpose = Column(Text, nullable=True)
    extra_context = Column(Text, nullable=True)
    llm_extra_comment = Column(Text, nullable=True)
    date_submitted = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_demo = Column(Boolean, default=False)

    # Scores
    ml_score = Column(Float, nullable=True)
    rule_score = Column(Float, nullable=True)
    anomaly_clean = Column(Integer, nullable=True)
    composite_score = Column(Float, nullable=True)
    tier = Column(String, nullable=True)

    # Rules breakdown
    rule_min_heads = Column(Integer, nullable=True)
    rule_norm_positive = Column(Integer, nullable=True)
    rule_amount_positive = Column(Integer, nullable=True)
    rule_not_extreme_amount = Column(Integer, nullable=True)
    rule_norm_in_range = Column(Integer, nullable=True)

    # SHAP explanations as JSON
    shap_json = Column(Text, nullable=True)

    # Decision
    decision = Column(String, default="pending")  # pending/approved/rejected/more_info
    decision_note = Column(Text, nullable=True)
    decision_by = Column(String, nullable=True)
    decision_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="applications")
    farmer_profile = relationship("FarmerProfile", back_populates="applications")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    _ensure_schema_columns()


def _ensure_schema_columns():
    inspector = inspect(engine)
    table_columns = {
        "farmer_profiles": {
            "annual_revenue": "FLOAT",
            "employee_count": "INTEGER",
            "owned_land_area_ha": "FLOAT",
            "leased_land_area_ha": "FLOAT",
            "lease_term_remaining_years": "INTEGER",
            "has_livestock_facility": "BOOLEAN",
            "has_feed_base": "BOOLEAN",
            "has_equipment": "BOOLEAN",
            "has_sales_contracts": "BOOLEAN",
            "has_tax_debt": "BOOLEAN",
            "has_overdue_loans": "BOOLEAN",
        },
        "applications": {
            "own_contribution_percent": "FLOAT",
            "planned_execution_months": "INTEGER",
            "has_supporting_documents": "BOOLEAN",
            "previous_subsidies_count": "INTEGER",
            "previous_subsidy_returns_count": "INTEGER",
            "previous_subsidy_violations_count": "INTEGER",
            "previous_subsidy_overdue_reports_count": "INTEGER",
            "previous_subsidies_closed_properly": "BOOLEAN",
            "requested_purpose": "TEXT",
            "extra_context": "TEXT",
            "llm_extra_comment": "TEXT",
        },
    }

    with engine.begin() as conn:
        for table_name, required_columns in table_columns.items():
            existing = {col["name"] for col in inspector.get_columns(table_name)}
            for column_name, column_type in required_columns.items():
                if column_name in existing:
                    continue
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))
