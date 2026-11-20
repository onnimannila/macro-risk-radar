"""Quantitative risk transformations and rolling statistics."""

from datetime import date
import logging
import re
import numpy as np
import pandas as pd
from morning_report.models import CalculatedMetric, SeriesDefinition


class TransformationEngine:
    @staticmethod
    def normalize_maturity(tenor_input: str | float) -> tuple[float, str]:
        """
        Normalizes provider maturity labels (6MO, 0.5Y, 1YR, 10Y, 30YR) to canonical
        numeric year values and display labels.
        """
        if isinstance(tenor_input, (int, float)):
            years = float(tenor_input)
            label = f"{int(years)}Y" if years >= 1.0 and years.is_integer() else f"{years}Y"
            return years, label

        raw = str(tenor_input).strip().upper()
        if raw in ["6M", "6MO", "0.5Y", "0.5YR", "6 MONTHS"]:
            return 0.5, "6M"
        if raw in ["3M", "3MO", "0.25Y"]:
            return 0.25, "3M"

        m = re.match(r"^(\d+(?:\.\d+)?)\s*(Y|YR|YEARS|YEAR|M|MO|MONTHS)?$", raw)
        if m:
            val = float(m.group(1))
            unit = m.group(2) or "Y"
            if unit in ["M", "MO", "MONTHS"]:
                return val / 12.0, f"{int(val)}M"
            return val, f"{int(val)}Y" if val.is_integer() else f"{val}Y"
        return 10.0, "10Y"

    @staticmethod
    def calculate_metrics(
        definition: SeriesDefinition,
        history_df: pd.DataFrame,
        target_date: date,
        min_history_zscore: int = 252,
        zscore_window: int = 756,
        vol_window: int = 60,
    ) -> CalculatedMetric:
        if history_df.empty:
            return CalculatedMetric(
                series_id=definition.id,
                metric_id=f"{definition.id}_METRIC",
                as_of_date=target_date,
                latest_value=None,
                change_unit=definition.unit,
                quality_status="missing",
                error_message="Series dataset is empty.",
                is_proxy=definition.is_proxy,
            )

        clean_df = history_df.dropna(subset=["observation_date", "value"]).copy()
        clean_df["observation_date"] = pd.to_datetime(clean_df["observation_date"]).dt.date
        clean_df = clean_df.drop_duplicates(subset=["observation_date"]).sort_values("observation_date")

        df = clean_df[clean_df["observation_date"] <= target_date]
        if df.empty:
            return CalculatedMetric(
                series_id=definition.id,
                metric_id=f"{definition.id}_METRIC",
                as_of_date=target_date,
                latest_value=None,
                change_unit=definition.unit,
                quality_status="missing",
                error_message="No observations on or before target date.",
                is_proxy=definition.is_proxy,
            )

        latest_row = df.iloc[-1]
        latest_val = float(latest_row["value"])
        latest_date = latest_row["observation_date"]
        stale_days = (target_date - latest_date).days

        # 1D Change
        prior_val, prior_date, change_val = None, None, None
        if len(df) >= 2:
            prev_row = df.iloc[-2]
            prior_val = float(prev_row["value"])
            prior_date = prev_row["observation_date"]
            if definition.transformation == "pct_change" and prior_val != 0.0:
                change_val = ((latest_val / prior_val) - 1.0) * 100.0
            elif definition.transformation == "diff_bp":
                diff = latest_val - prior_val
                change_val = diff if (definition.is_in_basis_points or definition.unit == "bp") else diff * 100.0
            else:
                change_val = latest_val - prior_val

        is_bp_diff = (definition.transformation == "diff_bp" and not definition.is_in_basis_points and definition.unit != "bp")
        # 5D Change (lag 5)
        pct_5d = None
        if len(df) >= 6:
            val_5d = float(df.iloc[-6]["value"])
            pct_5d = ((latest_val / val_5d) - 1.0) * 100.0 if definition.transformation == "pct_change" else (latest_val - val_5d) * (100.0 if is_bp_diff else 1.0)

        # 1M Change (~21 trading days)
        pct_1m = None
        if len(df) >= 22:
            val_1m = float(df.iloc[-22]["value"])
            pct_1m = ((latest_val / val_1m) - 1.0) * 100.0 if definition.transformation == "pct_change" else (latest_val - val_1m) * (100.0 if is_bp_diff else 1.0)

        # YTD Change
        pct_ytd = None
        prev_year_end = df[df["observation_date"] <= date(target_date.year - 1, 12, 31)]
        if not prev_year_end.empty:
            val_ytd = float(prev_year_end.iloc[-1]["value"])
            pct_ytd = ((latest_val / val_ytd) - 1.0) * 100.0 if definition.transformation == "pct_change" else (latest_val - val_ytd) * (100.0 if is_bp_diff else 1.0)

        # Rolling 60D Volatility (Annualized log returns for percentage transformations)
        vol_60d = None
        if len(df) >= vol_window + 1 and definition.transformation == "pct_change":
            vals = df["value"].tail(vol_window + 1).to_numpy(dtype=float)
            if (vals > 0).all():
                log_rets = np.log(vals[1:] / vals[:-1])
                vol_60d = float(np.std(log_rets, ddof=1) * np.sqrt(252) * 100.0)

        # 1Y Percentile: exactly last 252 valid observations
        percentile_1y = None
        tail_1y = df["value"].tail(252).to_numpy(dtype=float)
        if len(tail_1y) >= min_history_zscore and len(tail_1y) > 0:
            percentile_1y = float((tail_1y <= latest_val).sum() / len(tail_1y) * 100.0)

        # 3Y Z-score: strictly requires min_history_zscore and positive variance. Never returns 0.0 on failure.
        zscore_3y = None
        tail_3y = df["value"].tail(zscore_window + 1).to_numpy(dtype=float)
        if len(tail_3y) >= min_history_zscore + 1 and prior_val is not None:
            if definition.transformation == "pct_change":
                moves = (tail_3y[1:] / tail_3y[:-1]) - 1.0
                latest_move = (latest_val / prior_val) - 1.0
            else:
                moves = np.diff(tail_3y)
                latest_move = latest_val - prior_val
            std_move = float(np.std(moves, ddof=1)) if len(moves) > 1 else float(np.std(moves))
            if std_move > 1e-7 and not np.isnan(std_move):
                zscore_3y = float((latest_move - np.mean(moves)) / std_move)

        # Quality status
        if stale_days > definition.stale_after_days:
            quality = "stale"
        elif len(df) < min_history_zscore:
            quality = "insufficient_history"
        else:
            quality = "ok"

        unit_str = "%" if definition.transformation == "pct_change" else ("bp" if definition.transformation == "diff_bp" else definition.unit)
        return CalculatedMetric(
            series_id=definition.id,
            metric_id=f"{definition.id}_METRIC",
            as_of_date=latest_date,
            prior_date=prior_date,
            latest_value=latest_val,
            prior_value=prior_val,
            change_value=change_val,
            change_unit=unit_str,
            pct_5d=pct_5d,
            pct_1m=pct_1m,
            pct_ytd=pct_ytd,
            volatility_60d=vol_60d,
            percentile_1y=percentile_1y,
            zscore_3y=zscore_3y,
            quality_status=quality,
            is_proxy=definition.is_proxy,
            stale_days=stale_days,
        )