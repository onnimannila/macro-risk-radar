"""Data quality, plausibility, and integrity checks."""

from datetime import date, timedelta
import numpy as np
import pandas as pd
from morning_report.config import ThresholdConfig
from morning_report.models import CurvePoint, SeriesDefinition, ValidationResult


class DataValidationService:
    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.plausibility = thresholds.plausibility

    def validate_series_data(self, definition: SeriesDefinition, df: pd.DataFrame, target_date: date) -> ValidationResult:
        warnings, errors = [], []
        if df.empty:
            errors.append(f"Series {definition.id} has no observations.")
            return ValidationResult(series_id=definition.id, passed=False, warnings=[], errors=errors)

        required_cols = {"observation_date", "value"}
        if not required_cols.issubset(df.columns):
            errors.append(f"Missing required columns in {definition.id}: {required_cols - set(df.columns)}")
            return ValidationResult(series_id=definition.id, passed=False, warnings=[], errors=errors)

        # Non-finite values
        if not np.isfinite(df["value"].to_numpy(dtype=float)).all():
            errors.append(f"Non-finite (NaN or Inf) observations detected in {definition.id}.")

        if df["observation_date"].duplicated().any():
            warnings.append(f"Duplicate observation dates detected in {definition.id}.")

        clean_df = df.dropna(subset=["observation_date", "value"]).sort_values("observation_date")
        if clean_df.empty:
            errors.append(f"No valid observations remain after dropping NaNs in {definition.id}.")
            return ValidationResult(series_id=definition.id, passed=False, warnings=warnings, errors=errors)

        latest_row = df.iloc[-1]
        latest_date = latest_row["observation_date"]
        latest_val = float(latest_row["value"])

        if definition.asset_class in ["equities", "volatility"] and latest_val <= 0:
            errors.append(f"Non-positive price/level ({latest_val}) for asset class {definition.asset_class}.")

        # Sensible calendar lag: weekend days do not artificially penalize Friday prints
        calendar_days = (target_date - latest_date).days
        effective_stale = calendar_days > definition.stale_after_days + (2 if target_date.weekday() in [0, 6] else 0)
        if effective_stale:
            warnings.append(f"Data is stale: latest date {latest_date} is {(target_date - latest_date).days} days old.")

        if len(df) >= 2:
            prev_val = float(df.iloc[-2]["value"])
            if definition.transformation == "pct_change" and prev_val > 0:
                pct = abs(((latest_val / prev_val) - 1.0) * 100.0)
                if definition.asset_class == "equities" and pct > self.plausibility.max_equity_daily_move_pct:
                    errors.append(f"Implausible equity jump of {pct:.1f}%.")
                elif definition.asset_class == "fx" and pct > self.plausibility.max_fx_daily_move_pct:
                    errors.append(f"Implausible FX jump of {pct:.1f}%.")
            elif definition.transformation == "diff_bp":
                diff = abs(latest_val - prev_val)
                bp = diff if definition.is_in_basis_points or definition.unit == "bp" else diff * 100.0
                if definition.asset_class == "rates" and bp > self.plausibility.max_rates_daily_move_bp:
                    errors.append(f"Implausible rates jump of {bp:.1f} bp.")
                elif definition.asset_class == "credit" and bp > self.plausibility.max_credit_daily_move_bp:
                    errors.append(f"Implausible credit jump of {bp:.1f} bp.")

        passed = len(errors) == 0
        return ValidationResult(series_id=definition.id, passed=passed, warnings=warnings, errors=errors)

    @staticmethod
    def validate_curve(points: list[CurvePoint], max_date_spread_days: int = 4) -> tuple[bool, list[str], list[str]]:
        warnings, errors = [], []
        if len(points) < 4:
            errors.append(f"Yield curve has only {len(points)} points; minimum 4 required for a connected curve.")
            return False, warnings, errors

        # Numeric tenor validation
        tenors = [p.tenor_years for p in points]
        if len(tenors) != len(set(tenors)):
            warnings.append("Duplicate tenors detected on curve snapshot.")

        has_short = any(t <= 2.0 for t in tenors)
        has_medium = any(3.0 <= t <= 7.0 for t in tenors)
        has_long = any(t >= 10.0 for t in tenors)

        if not has_short:
            warnings.append("Curve lacks a short-end tenor (<= 2Y).")
        if not has_medium:
            warnings.append("Curve lacks a belly/medium tenor (3Y - 7Y).")
        if not has_long:
            warnings.append("Curve lacks a long-end tenor (>= 10Y).")

        # Date tolerance check
        dates = [p.observation_date for p in points]
        min_d, max_d = min(dates), max(dates)
        if (max_d - min_d).days > max_date_spread_days:
            warnings.append(f"Curve points span { (max_d - min_d).days } days ({min_d} to {max_d}).")

        # Unreasonable yields
        if any(p.yield_pct < -2.0 or p.yield_pct > 30.0 for p in points):
            errors.append("Curve contains implausible yield values outside [-2%, 30%].")

        return len(errors) == 0, warnings, errors