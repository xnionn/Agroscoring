"""
Dataset Service — loads and provides access to real agricultural datasets from Downloads
for integration into the scoring system. Provides benchmarks and context for better scoring.
"""
import os
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

DOWNLOADS = r"C:\Users\Мади\Downloads"


class DatasetService:
    """
    Singleton service to load CSV, Excel, and JSON datasets from Downloads
    and provide methods for scoring system integration.
    """
    
    def __init__(self):
        self.subsidy_data = None  # "Выгрузка по выданным субсидиям" (granted subsidies)
        self.facility_registry = None  # "reestr_uboihraneniepererab" (facility registry)
        self.farmer_registry = None  # "shko_sharua_zane_fermerlik_ko" (farmer registry)
        self.granularity_map = None  # --Granularity-Joinkeys.csv (metadata)
        self._loaded = False

    def load(self):
        """Load all available datasets from Downloads."""
        if self._loaded:
            return
        
        logger.info("[DatasetService] Starting data loading...")
        
        self._load_subsidy_data()
        self._load_facility_registry()
        self._load_farmer_registry()
        self._load_granularity_map()
        
        self._loaded = True
        logger.info("[DatasetService] All datasets loaded OK")

    def _load_subsidy_data(self):
        """Load granted subsidy data for regional benchmarking."""
        path = os.path.join(DOWNLOADS, "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx")
        try:
            df = pd.read_excel(path, sheet_name="Page 1")
            self.subsidy_data = df
            logger.info(f"[DatasetService] Subsidy data loaded: {df.shape[0]} rows, {df.shape[1]} cols")
        except Exception as e:
            logger.warning(f"[DatasetService] Could not load subsidy data: {e}")
            self.subsidy_data = None

    def _load_facility_registry(self):
        """Load facility registry (reestр производителей и переработчиков)."""
        # Try JSON format first
        json_path = os.path.join(DOWNLOADS, "reestr_uboihraneniepererab-v10.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    self.facility_registry = pd.DataFrame(data)
                    logger.info(f"[DatasetService] Facility registry (JSON) loaded: {len(data)} facilities")
                    return
            except Exception as e:
                logger.warning(f"[DatasetService] Could not load facility registry JSON: {e}")

    def _load_farmer_registry(self):
        """Load farmer registry for family farms context."""
        json_path = os.path.join(DOWNLOADS, "shko_sharua_zane_fermerlik_ko.csv")
        if os.path.exists(json_path):
            try:
                df = pd.read_csv(json_path)
                self.farmer_registry = df
                logger.info(f"[DatasetService] Farmer registry loaded: {df.shape[0]} farmers")
            except Exception as e:
                logger.warning(f"[DatasetService] Could not load farmer registry: {e}")

    def _load_granularity_map(self):
        """Load granularity and join keys metadata."""
        path = os.path.join(DOWNLOADS, "--Granularity-Joinkeys.csv")
        try:
            df = pd.read_csv(path)
            self.granularity_map = df
            logger.info(f"[DatasetService] Granularity map loaded: {df.shape[0]} datasets")
        except Exception as e:
            logger.warning(f"[DatasetService] Could not load granularity map: {e}")

    # === Public API ===

    def get_region_subsidy_context(self, region: str) -> Dict:
        """
        Get subsidy context for a specific region (e.g., number of subsidies, avg amount).
        Used for regional benchmarking.
        """
        if self.subsidy_data is None or self.subsidy_data.empty:
            return {}

        try:
            # Filter by region (adapt column name if needed)
            region_data = self.subsidy_data[self.subsidy_data.get('region') == region]
            if region_data.empty:
                return {}

            return {
                "region": region,
                "total_subsidies": len(region_data),
                "avg_subsidy_amount": float(region_data.get('amount', pd.Series()).mean()) if 'amount' in region_data.columns else 0,
                "total_amount": float(region_data.get('amount', pd.Series()).sum()) if 'amount' in region_data.columns else 0,
            }
        except Exception as e:
            logger.warning(f"[DatasetService] Error getting region context for {region}: {e}")
            return {}

    def get_direction_norms(self, direction: str) -> Dict:
        """
        Get normatиve data (norm per head) for a livestock direction.
        Helps validate if a requested norm is realistic.
        """
        if self.subsidy_data is None or self.subsidy_data.empty:
            return {}

        try:
            # Filter by direction
            direction_data = self.subsidy_data[self.subsidy_data.get('direction') == direction]
            if direction_data.empty:
                return {}

            if 'norm' in direction_data.columns:
                norms = direction_data['norm'].dropna()
                return {
                    "direction": direction,
                    "min_norm": float(norms.min()),
                    "max_norm": float(norms.max()),
                    "mean_norm": float(norms.mean()),
                    "median_norm": float(norms.median()),
                    "count": len(norms),
                }
        except Exception as e:
            logger.warning(f"[DatasetService] Error getting direction norms for {direction}: {e}")
            return {}

    def get_facility_by_region(self, region: str) -> Optional[pd.DataFrame]:
        """Get all facilities in a specific region from the registry."""
        if self.facility_registry is None or self.facility_registry.empty:
            return None

        try:
            region_facilities = self.facility_registry[
                self.facility_registry.get('region') == region
            ]
            return region_facilities if not region_facilities.empty else None
        except Exception as e:
            logger.warning(f"[DatasetService] Error filtering facilities: {e}")
            return None

    def validate_norm_reasonableness(self, direction: str, requested_norm: float) -> Dict:
        """
        Check if a requested norm is within 'reasonable' bounds for the direction.
        Returns comparison to historical data.
        """
        norms = self.get_direction_norms(direction)
        if not norms or norms.get('count', 0) == 0:
            return {"valid": True, "reason": "No historical data to compare"}

        mean = norms.get('mean_norm', requested_norm)
        median = norms.get('median_norm', requested_norm)

        # Check if within 2σ of median
        if mean > 0:
            ratio = requested_norm / median
            if 0.5 <= ratio <= 2.0:
                return {
                    "valid": True,
                    "reason": f"Norm {requested_norm} is within reasonable range (median {median:.0f})",
                    "percentile": self._calc_percentile(requested_norm, norms),
                }
            else:
                return {
                    "valid": False,
                    "reason": f"Norm {requested_norm} deviates strongly from median {median:.0f} (ratio {ratio:.2f})",
                    "percentile": self._calc_percentile(requested_norm, norms),
                }

        return {"valid": True, "reason": "Insufficient data for validation"}

    def _calc_percentile(self, value: float, norms_dict: Dict) -> Optional[float]:
        """Calculate which percentile a value falls into."""
        try:
            if self.subsidy_data is None or self.subsidy_data.empty:
                return None
            all_norms = self.subsidy_data['norm'].dropna()
            if all_norms.empty:
                return None
            percentile = (all_norms < value).sum() / len(all_norms) * 100
            return round(percentile, 1)
        except Exception:
            return None

    def get_region_approval_trends(self, region: str, lookback_months: int = 12) -> Dict:
        """
        Get approval trends for a region (if data has dates).
        Usefull for scoring which can be season-dependent.
        """
        if self.subsidy_data is None:
            return {}
        # Placeholder: would require date parsing
        return {"region": region, "trend": "data_not_yet_parsed"}

    def export_dataset_metadata(self) -> Dict:
        """
        Return metadata about all loaded datasets for system documentation.
        """
        return {
            "subsidy_data": {
                "loaded": self.subsidy_data is not None,
                "rows": len(self.subsidy_data) if self.subsidy_data is not None else 0,
                "file": "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx"
            },
            "facility_registry": {
                "loaded": self.facility_registry is not None,
                "rows": len(self.facility_registry) if self.facility_registry is not None else 0,
                "file": "reestr_uboihraneniepererab-v10.json"
            },
            "farmer_registry": {
                "loaded": self.farmer_registry is not None,
                "rows": len(self.farmer_registry) if self.farmer_registry is not None else 0,
                "file": "shko_sharua_zane_fermerlik_ko.csv"
            },
            "granularity_map": {
                "loaded": self.granularity_map is not None,
                "rows": len(self.granularity_map) if self.granularity_map is not None else 0,
                "file": "--Granularity-Joinkeys.csv"
            },
        }


# Singleton
dataset_service = DatasetService()
