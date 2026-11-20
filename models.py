"""Pydantic schemas and domain data models."""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field


class SurfaceMode(str, Enum):
    MARKET_IMPLIED = "MARKET_IMPLIED"
    UNAVAILABLE = "UNAVAILABLE"


class SurfaceStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    AVAILABLE_WITH_WARNINGS = "AVAILABLE_WITH_WARNINGS"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class RegimeClassification(BaseModel):
    regime_type: str = "Cross-Asset Regime"
    horizon: str = "1D"
    label: str
    plain_language_definition: str
    score: float | None = None
    confidence: str = "HIGH"
    evidence: list[str] = Field(default_factory=list)
    component_contributions: dict[str, float | None] = Field(default_factory=dict)
    previous_label: str | None = None
    regime_start_date: date | None = None
    duration_sessions: int = 1
    frequency_1m: float | None = None
    frequency_3m: float | None = None
    frequency_1y: float | None = None
    percentile_1y: float | None = None
    zscore_3y: float | None = None
    warnings: list[str] = Field(default_factory=list)
    status: str = "OK"


class YieldCurveRegime(BaseModel):
    label: str
    plain_language_definition: str
    short_move_bp: float | None = None
    long_move_bp: float | None = None
    slope_change_bp: float | None = None
    horizon_labels: dict[str, str] = Field(default_factory=dict)
    duration_sessions: int = 1
    frequency_1y: float | None = None
    risk_implication: str = ""
    evidence: list[str] = Field(default_factory=list)
    status: str = "OK"


class GraphSourceMetadata(BaseModel):
    provider_name: str
    series_id: str
    instrument: str
    effective_date: str
    retrieval_timestamp: str
    cache_status: str = "MISS"
    fallback_used: bool = False
    transformation: str = "level"
    unit: str = ""

class SeriesDefinition(BaseModel):
    id: str
    display_name: str
    asset_class: Literal["rates", "equities", "credit", "fx", "commodities", "volatility", "systemic_risk", "macro"] = "rates"
    provider: Literal["ecb", "fred", "eurostat", "market_data", "licensed", "local_csv", "synthetic"] = "ecb"
    provider_series_id: str
    unit: str
    currency: str = "EUR"
    frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "daily"
    transformation: Literal["pct_change", "diff_bp", "diff_level", "level"] = "pct_change"
    country: str | None = None
    tenor: str | None = None
    tenor_years: float | None = None
    curve_group: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    secondary_provider: str | None = None
    price_field: str | None = None
    proxy_description: str | None = None
    expected_release_lag: int = 1
    stale_after_days: int = 3
    is_in_basis_points: bool = False
    higher_is_riskier: bool = False
    is_proxy: bool = False
    enabled: bool = True


class CanonicalSeriesResult(BaseModel):
    """Canonical container for observation history and execution metadata."""
    series_id: str
    display_name: str
    asset_class: str
    country: Optional[str] = None
    currency: str = "EUR"
    tenor: Optional[str] = None
    unit: str
    provider_requested: str
    provider_used: Optional[str] = None
    fallback_provider: Optional[str] = None
    retrieval_status: str = "PENDING"
    retrieval_error: Optional[str] = None
    raw_observation_count: int = 0
    valid_observation_count: int = 0
    history: Any = Field(default_factory=pd.DataFrame)
    first_observation_date: Optional[date] = None
    latest_observation_date: Optional[date] = None
    requested_report_date: date
    effective_observation_date: Optional[date] = None
    age_calendar_days: int = 0
    age_business_days: int = 0
    stale: bool = False
    stale_after_days: int = 3
    expected_release_lag: int = 1
    cache_status: str = "MISS"
    synthetic: bool = False
    validation_status: str = "PENDING"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CalculatedMetric(BaseModel):
    series_id: str
    metric_id: str
    as_of_date: date
    prior_date: date | None = None
    latest_value: float | None = None
    prior_value: float | None = None
    change_value: float | None = None
    change_unit: str
    pct_5d: float | None = None
    pct_1m: float | None = None
    pct_ytd: float | None = None
    volatility_60d: float | None = None
    percentile_1y: float | None = None
    zscore_3y: float | None = None
    quality_status: Literal["ok", "stale", "insufficient_history", "warning", "missing", "error"] = "ok"
    error_message: str | None = None
    is_proxy: bool = False
    stale_days: int = 0


class TableRowItem(BaseModel):
    name: str
    level: str
    raw_level: float | None = None
    unit: str = ""
    chg: str
    chg_val: float | None = None
    chg_5d: str
    chg_1m: str = "-"
    chg_ytd: str = "-"
    pctile: str
    vol: str = "-"
    zscore: str = "-"
    as_of: str = "-"
    status: str = "OK"
    is_adverse: bool = False
    is_proxy: bool = False
    country: str | None = None
    tenor: str | None = None
    provider: str = "-"
    obs_date: str = "-"
    methodology: str = ""


class SovereignRateRow(BaseModel):
    country_code: str
    country_name: str
    currency: str
    yield_2y: str = "-"
    yield_5y: str = "-"
    yield_10y: str = "-"
    yield_30y: str = "-"
    yield_50y: str = "-"
    spread_2s10s: str = "-"
    spread_5s30s: str = "-"
    chg_10y_1d: str = "-"
    chg_10y_1d_val: float | None = None
    chg_10y_5d: str = "-"
    chg_10y_1m: str = "-"
    zscore_10y_3y: str = "-"
    as_of: str = "-"
    data_age_days: int = 0
    provider: str = "-"
    status: str = "OK"
    notes: str = ""


class CurvePoint(BaseModel):
    tenor: str
    tenor_years: float
    yield_pct: float
    observation_date: date
    instrument_id: str


class CurveSnapshot(BaseModel):
    curve_id: str
    display_name: str
    country: str
    currency: str
    as_of_date: date
    points: list[CurvePoint]
    is_complete: bool = True
    warnings: list[str] = Field(default_factory=list)


class SeriesQualityItem(BaseModel):
    series_id: str
    display_name: str
    provider: str
    status: str
    obs_count: int
    latest_date: str
    stale_days: int
    is_demo: bool = False
    retrieval_status: str = "RETRIEVED"
    error_detail: str | None = None
    missing_pct: float = 0.0
    raw_count: int = 0
    first_date: str = "-"
    effective_date: str = "-"
    cache_status: str = "MISS"
    synthetic: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class NewsItem(BaseModel):
    title: str
    publisher: str
    published_at: datetime
    url: str
    category: str
    relevance_score: float = 0.0


class OptionMarketObservation(BaseModel):
    """Canonical option observation from exchange market data."""
    provider: str = "DERIBIT"
    underlying: str = "BTC"
    instrument_name: str
    option_type: Literal["call", "put"]
    strike: float
    expiry_timestamp: datetime
    maturity_years: float
    days_to_expiry: float
    underlying_price: float
    forward_price: float
    log_forward_moneyness: float
    forward_moneyness: float
    mark_price: float | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    mark_iv: float | None = None  # Stored in annualized decimal form [0.05, 3.00]
    bid_iv: float | None = None
    ask_iv: float | None = None
    open_interest: float = 0.0
    volume: float = 0.0
    retrieval_timestamp: datetime
    data_timestamp: datetime | None = None
    quality_status: str = "VALID"
    exclusion_reason: str | None = None


class BTCVolatilitySurfaceResult(BaseModel):
    """Canonical container for BTC/USD market-implied SVI volatility surface."""
    surface_type: SurfaceMode = SurfaceMode.MARKET_IMPLIED
    provider: str = "DERIBIT"
    underlying: str = "BTC"
    quote_currency: str = "USD"
    valuation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valuation_timestamp_helsinki: str = ""
    spot_or_index_price: float = 0.0
    raw_instrument_count: int = 0
    valid_observation_count: int = 0
    retained_expiries: list[str] = Field(default_factory=list)
    retained_expiry_maturities: list[float] = Field(default_factory=list)
    retained_quote_count: int = 0
    excluded_quote_count: int = 0
    exclusions_by_reason: dict[str, int] = Field(default_factory=dict)
    calls_retained: int = 0
    puts_retained: int = 0
    average_open_interest: float = 0.0
    average_bid_ask_spread_iv: float = 0.0
    # Direct exchange observation lists (for plotting distinct markers)
    observed_expiries_years: list[float] = Field(default_factory=list)
    observed_log_moneyness: list[float] = Field(default_factory=list)
    observed_forward_moneyness: list[float] = Field(default_factory=list)
    observed_volatilities_pct: list[float] = Field(default_factory=list)
    observed_option_types: list[str] = Field(default_factory=list)
    # Dense calibrated model evaluations (SVI evaluation grid, NOT exchange observations)
    dense_maturity_grid: list[float] = Field(default_factory=list)
    dense_moneyness_grid: list[float] = Field(default_factory=list)
    dense_forward_moneyness_grid: list[float] = Field(default_factory=list)
    dense_vol_matrix: list[list[float | None]] = Field(default_factory=list)
    calibration_method: str = "Raw SVI per Expiry with Monotone Spline Variance Interpolation"
    calibration_parameters: dict[str, Any] = Field(default_factory=dict)
    calibration_rmse_by_expiry: dict[str, float] = Field(default_factory=dict)
    weighted_rmse: float | None = None
    direct_observation_count: int = 0
    dense_model_point_count: int = 0
    dense_grid_coverage_pct: float = 0.0
    arbitrage_diagnostics: dict[str, Any] = Field(default_factory=dict)
    liquidity_diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: SurfaceStatus = SurfaceStatus.UNAVAILABLE


class ValidationResult(BaseModel):
    series_id: str
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DataQualitySummary(BaseModel):
    total_series: int = 0
    valid_series: int = 0
    stale_series: int = 0
    failed_series: int = 0
    missing_series: int = 0
    missing_api_keys: int = 0
    source_errors: int = 0
    insufficient_history: int = 0
    demo_mode_active: bool = False
    items: list[SeriesQualityItem] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.items)


class RelativeValueSpread(BaseModel):
    label: str
    spread_bp: str
    spread_val: float | None = None
    chg_1d: str = "-"
    chg_5d: str = "-"
    chg_1m: str = "-"
    long_date: str = "-"
    benchmark_date: str = "-"
    date_diff_days: int = 0
    provider: str = "-"
    quality_status: str = "OK"
    commentary: str


class ReportContext(BaseModel):
    as_of_date: str
    execution_time: str
    timezone: str
    run_status: str
    headline: str = "Morning Macro-Financial Risk Monitor"
    regime_label: str
    curve_label: str
    max_rates_move: str
    stale_count: int
    today_market_summary: list[str] = Field(default_factory=list)
    yield_curve_regime: YieldCurveRegime | None = None
    cross_asset_regime_detail: RegimeClassification | None = None
    missing_count: int = 0
    failed_count: int = 0
    charts_rendered_count: int = 0
    missing_api_keys: int = 0
    source_errors: int = 0
    insufficient_history_count: int = 0
    release_lag_summary: str = ""
    eurusd_chart_uri: str | None = None
    vix_chart_uri: str | None = None
    sovereign_curve_interpretation: str = ""
    rates_history_interpretation: str = ""
    credit_history_interpretation: str = ""
    eurusd_interpretation: str = ""
    vix_interpretation: str = ""
    zscore_interpretation: str = ""
    btc_surface_interpretation: str = ""
    lag_recommendation: str = ""
    demo_mode: bool = False
    sovereign_rates_table: list[SovereignRateRow] = Field(default_factory=list)
    relative_value_spreads: list[RelativeValueSpread] = Field(default_factory=list)
    rates_table: list[TableRowItem] = Field(default_factory=list)
    market_table: list[TableRowItem] = Field(default_factory=list)
    sovereign_curve_chart_uri: str | None = None
    rates_history_chart_uri: str | None = None
    credit_history_chart_uri: str | None = None
    zscore_chart_uri: str | None = None
    chart_diagnostics: dict[str, str] = Field(default_factory=dict)
    narrative_bullets: list[str] = Field(default_factory=list)
    news_items: list[dict[str, Any]] = Field(default_factory=list)
    quality_summary: DataQualitySummary = Field(default_factory=DataQualitySummary)
    curve_snapshots: list[CurveSnapshot] = Field(default_factory=list)
    data_warnings: list[str] = Field(default_factory=list)
    btc_volatility_surface: BTCVolatilitySurfaceResult | None = None
    vol_surface_chart_uri: str | None = None
    vol_surface_3d_chart_uri: str | None = None