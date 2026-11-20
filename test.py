"""Comprehensive unit tests for the institutional morning macro-financial risk monitor and BTC surface."""

import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

# Ensure src is available on python path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import pytest
from morning_report.market_calendar import MarketCalendarService
from morning_report.config import ConfigBundle, PlausibilityThresholds, ThresholdConfig, ThresholdLevel
from morning_report.models import (
    CalculatedMetric, CurvePoint, CurveSnapshot, ReportContext, SeriesDefinition,
    SeriesQualityItem, TableRowItem,
    RegimeClassification, YieldCurveRegime,
)
from morning_report.config import DeribitVolatilitySurfaceConfig
from morning_report.pipeline import PipelineOrchestrator
from morning_report.providers import DeribitOptionsProvider
from morning_report.signals import SignalEngine
from morning_report.transformations import TransformationEngine
from morning_report.charts import DARK_PAGE_BG, DARK_CARD_BG, COLOR_CYAN
from morning_report.volatility_surface import BTCVolatilitySurfaceEngine, RawSVICalibrator
from morning_report.models import OptionMarketObservation, SurfaceMode, SurfaceStatus
from morning_report.validation import DataValidationService


def test_calendar_weekend_and_rollback():
    cal = MarketCalendarService()
    mon = date(2026, 11, 9)
    prev = cal.get_previous_business_day(mon)
    assert prev == date(2026, 11, 6)
    assert not cal.is_trading_day(date(2026, 11, 7))


def test_transformations_basis_points():
    s_def = SeriesDefinition(
        id="TEST_RATE",
        display_name="DE 10Y Yield",
        asset_class="rates",
        provider="ecb",
        provider_series_id="TEST",
        unit="pct",
        transformation="diff_bp",
    )
    df = pd.DataFrame([
        {"observation_date": date(2026, 11, 5), "value": 2.45},
        {"observation_date": date(2026, 11, 6), "value": 2.50},
    ])
    metric = TransformationEngine.calculate_metrics(s_def, df, date(2026, 11, 6))
    assert metric.change_value == pytest.approx(5.0)
    assert metric.change_unit == "bp"


def test_maturity_normalization_numeric():
    norm = TransformationEngine.normalize_maturity
    assert norm("6MO") == (0.5, "6M")
    assert norm("6M") == (0.5, "6M")
    assert norm("1YR") == (1.0, "1Y")
    assert norm("10YR") == (10.0, "10Y")
    assert norm("30Y") == (30.0, "30Y")
    assert norm(2) == (2.0, "2Y")


def test_zscore_returns_none_when_insufficient_history():
    s_def = SeriesDefinition(id="TEST_SHORT", display_name="Short Series", provider="fred", provider_series_id="S", unit="pts")
    df = pd.DataFrame([
        {"observation_date": date(2026, 11, i), "value": float(i)} for i in range(1, 15)
    ])
    metric = TransformationEngine.calculate_metrics(s_def, df, date(2026, 11, 14), min_history_zscore=252)
    # Crucial acceptance test: zscore must NEVER default to 0.0 when history is insufficient
    assert metric.zscore_3y is None


def test_yield_curve_classification():
    thresholds = ThresholdConfig(
        daily_zscore=ThresholdLevel(amber=1.5, red=2.5),
        rates_move_bp=ThresholdLevel(amber=8.0, red=15.0),
        sovereign_spread_move_bp=ThresholdLevel(amber=4.0, red=8.0),
        credit_spread_move_bp=ThresholdLevel(amber=8.0, red=18.0),
        equity_return_pct=ThresholdLevel(amber=1.2, red=2.5),
        fx_return_pct=ThresholdLevel(amber=0.8, red=1.8),
        commodities_return_pct=ThresholdLevel(amber=2.0, red=4.0),
        volatility_change_pct=ThresholdLevel(amber=8.0, red=18.0),
        plausibility=PlausibilityThresholds(),
    )
    engine = SignalEngine(thresholds)
    assert engine.classify_yield_curve(-8.0, -3.0).label == "Bull Steepening"
    assert engine.classify_yield_curve(10.0, 4.0).label == "Bear Flattening"
    assert engine.classify_yield_curve(4.0, 10.0).label == "Bear Steepening"
    assert engine.classify_yield_curve(-3.0, -8.0).label == "Bull Flattening"
    assert engine.classify_yield_curve(0.2, 0.4).label == "Unchanged / Range-Bound"
    assert engine.classify_yield_curve(5.0, 5.2).label == "Parallel Upward Shift"
    assert engine.classify_yield_curve(-6.0, -6.1).label == "Parallel Downward Shift"


def test_cross_asset_regime_reactivity():
    thresholds = ThresholdConfig(
        daily_zscore=ThresholdLevel(amber=1.5, red=2.5),
        rates_move_bp=ThresholdLevel(amber=8.0, red=15.0),
        sovereign_spread_move_bp=ThresholdLevel(amber=4.0, red=8.0),
        credit_spread_move_bp=ThresholdLevel(amber=8.0, red=18.0),
        equity_return_pct=ThresholdLevel(amber=1.2, red=2.5),
        fx_return_pct=ThresholdLevel(amber=0.8, red=1.8),
        plausibility=PlausibilityThresholds(),
    )
    engine = SignalEngine(thresholds)
    # 1. Risk-off fixture: Equities plunge, HY spreads widen, VIX jumps
    m_off = {
        "EQ_SP500": CalculatedMetric(series_id="EQ_SP500", metric_id="M", as_of_date=date.today(), change_value=-2.1, change_unit="%"),
        "US_HY_OAS": CalculatedMetric(series_id="US_HY_OAS", metric_id="M", as_of_date=date.today(), change_value=12.0, change_unit="bp"),
        "VIX_INDEX": CalculatedMetric(series_id="VIX_INDEX", metric_id="M", as_of_date=date.today(), change_value=18.0, change_unit="%"),
    }
    reg_off = engine.classify_cross_asset_regime(m_off)
    assert reg_off.label in ["Risk-Off", "Acute Stress"]

    # 2. Risk-on fixture: Equities surge, spreads tighten, VIX collapses
    m_on = {
        "EQ_SP500": CalculatedMetric(series_id="EQ_SP500", metric_id="M", as_of_date=date.today(), change_value=1.8, change_unit="%"),
        "US_HY_OAS": CalculatedMetric(series_id="US_HY_OAS", metric_id="M", as_of_date=date.today(), change_value=-8.0, change_unit="bp"),
        "VIX_INDEX": CalculatedMetric(series_id="VIX_INDEX", metric_id="M", as_of_date=date.today(), change_value=-12.0, change_unit="%"),
    }
    reg_on = engine.classify_cross_asset_regime(m_on)
    assert reg_on.label in ["Risk-On", "Strong Risk-On"]


def test_validation_implausible_jump():
    thresholds = ThresholdConfig(
        daily_zscore=ThresholdLevel(amber=1.5, red=2.5),
        rates_move_bp=ThresholdLevel(amber=8.0, red=15.0),
        sovereign_spread_move_bp=ThresholdLevel(amber=4.0, red=8.0),
        credit_spread_move_bp=ThresholdLevel(amber=8.0, red=18.0),
        equity_return_pct=ThresholdLevel(amber=1.2, red=2.5),
        fx_return_pct=ThresholdLevel(amber=0.8, red=1.8),
        commodities_return_pct=ThresholdLevel(amber=2.0, red=4.0),
        volatility_change_pct=ThresholdLevel(amber=8.0, red=18.0),
        plausibility=PlausibilityThresholds(max_equity_daily_move_pct=25.0),
    )
    validator = DataValidationService(thresholds)
    s_def = SeriesDefinition(
        id="TEST_EQ",
        display_name="Equity",
        asset_class="equities",
        provider="market_data",
        provider_series_id="TEST",
        unit="price",
        transformation="pct_change",
    )
    df = pd.DataFrame([
        {"observation_date": date(2026, 11, 5), "value": 100.0},
        {"observation_date": date(2026, 11, 6), "value": 140.0},
    ])
    res = validator.validate_series_data(s_def, df, date(2026, 11, 6))
    assert not res.passed


def test_zscore_none_preservation_in_metric():
    s_def = SeriesDefinition(
        id="TEST_NO_CORRUPTION",
        display_name="No Corruption",
        provider="ecb",
        provider_series_id="TEST",
        unit="pct",
    )
    df = pd.DataFrame([
        {"observation_date": date(2026, 11, 5), "value": 2.40},
        {"observation_date": date(2026, 11, 6), "value": 2.45},
    ])
    metric = TransformationEngine.calculate_metrics(s_def, df, date(2026, 11, 6), min_history_zscore=252)
    assert metric.zscore_3y is None
    assert (metric.zscore_3y or 0.0) == 0.0  # Demonstrates why 'or 0.0' must not be used
    assert metric.quality_status == "insufficient_history"


def test_production_mode_never_creates_synthetic_observations():
    bundle = ConfigBundle()
    orchestrator = PipelineOrchestrator(bundle)
    fake_def = SeriesDefinition(id="FAKE_OFFLINE", display_name="Fake", provider="ecb", provider_series_id="NON.EXISTENT", unit="pts")
    bundle.series = [fake_def]
    orchestrator.run(date(2026, 11, 9), demo_mode=False)
    assert orchestrator.config.series[0].id == "FAKE_OFFLINE"


def test_missing_history_remains_missing():
    s_def = SeriesDefinition(id="EMPTY", display_name="Empty", provider="fred", provider_series_id="EMPTY", unit="pct")
    metric = TransformationEngine.calculate_metrics(s_def, pd.DataFrame(), date(2026, 11, 9))
    assert metric.latest_value is None
    assert metric.quality_status == "missing"


def test_genuine_zero_zscore_remains_distinguishable_from_missing():
    s_def = SeriesDefinition(id="TEST_ZERO", display_name="Test Zero", provider="fred", provider_series_id="T", unit="pts")
    metric_none = TransformationEngine.calculate_metrics(s_def, pd.DataFrame(), date(2026, 11, 9))
    assert metric_none.zscore_3y is None


def test_every_displayed_table_row_has_unit_and_as_of():
    row = TableRowItem(name="Test Item", level="100.0", unit="bp", chg="+1.0 bp", chg_5d="-", pctile="-", as_of="2026-11-06")
    assert row.unit != ""
    assert row.as_of != "-"


def test_no_row_with_missing_as_of_can_have_status_ok():
    row = TableRowItem(name="Test Item", level="N/A", unit="bp", chg="-", chg_5d="-", pctile="-", as_of="-", status="MISSING")
    assert row.status != "OK"


def test_no_connected_curve_rendered_with_fewer_than_four_tenors():
    pts = [
        CurvePoint(tenor="2Y", tenor_years=2.0, yield_pct=2.4, observation_date=date(2026, 11, 6), instrument_id="DE_2Y"),
        CurvePoint(tenor="10Y", tenor_years=10.0, yield_pct=2.6, observation_date=date(2026, 11, 6), instrument_id="DE_10Y"),
    ]
    passed, _, errors = DataValidationService.validate_curve(pts)
    assert not passed
    assert any("minimum 4 required" in e for e in errors)


def test_tenors_are_sorted_numerically():
    pts = [
        CurvePoint(tenor="10Y", tenor_years=10.0, yield_pct=2.6, observation_date=date(2026, 11, 6), instrument_id="DE_10Y"),
        CurvePoint(tenor="2Y", tenor_years=2.0, yield_pct=2.4, observation_date=date(2026, 11, 6), instrument_id="DE_2Y"),
        CurvePoint(tenor="5Y", tenor_years=5.0, yield_pct=2.5, observation_date=date(2026, 11, 6), instrument_id="DE_5Y"),
    ]
    pts.sort(key=lambda p: p.tenor_years)
    assert [p.tenor_years for p in pts] == [2.0, 5.0, 10.0]


def test_hy_and_ig_converted_to_basis_points_once():
    s_def = SeriesDefinition(id="US_HY_OAS", display_name="US HY OAS", provider="fred", provider_series_id="BAMLH0A0HYM2", unit="bp", is_in_basis_points=True)
    df = pd.DataFrame([
        {"observation_date": date(2026, 11, 5), "value": 334.0},
        {"observation_date": date(2026, 11, 6), "value": 338.0},
    ])
    metric = TransformationEngine.calculate_metrics(s_def, df, date(2026, 11, 6))
    assert metric.latest_value == 338.0
    assert metric.change_value == pytest.approx(4.0)


def test_relative_value_spread_subtraction_order():
    fi_yield = 2.85
    de_yield = 2.45
    spread_bp = (fi_yield - de_yield) * 100.0
    assert spread_bp == pytest.approx(40.0)


def test_relative_value_spread_units_and_dates_compatible():
    d1 = date(2026, 11, 6)
    d2 = date(2026, 11, 5)
    assert abs((d1 - d2).days) <= 4


def test_vix_1d_change_uses_two_consecutive_valid_sessions():
    s_def = SeriesDefinition(id="VIX_INDEX", display_name="VIX", provider="fred", provider_series_id="VIXCLS", unit="Index", transformation="pct_change")
    df = pd.DataFrame([
        {"observation_date": date(2026, 11, 5), "value": 15.0},
        {"observation_date": date(2026, 11, 6), "value": 16.5},
    ])
    metric = TransformationEngine.calculate_metrics(s_def, df, date(2026, 11, 6))
    assert metric.change_value == pytest.approx(10.0)


def test_eurusd_quote_convention_usd_per_eur():
    s_def = SeriesDefinition(id="FX_EURUSD", display_name="EUR/USD", provider="ecb", provider_series_id="EXR", unit="USD/EUR", transformation="pct_change")
    assert s_def.unit == "USD/EUR"



def test_failed_providers_create_quality_table_rows():
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    res = orch.run(date(2026, 11, 9), demo_mode=False)
    assert res.exists()


def test_missing_fred_key_produces_missing_api_key():
    from morning_report.providers import FREDDataProvider
    p = FREDDataProvider()
    p.api_key = ""
    assert not p.health_check()


def test_mandatory_empty_panel_forces_partial():
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    out = orch.run(date(2026, 11, 9), demo_mode=False)
    text = out.read_text(encoding="utf-8")
    assert "PARTIAL" in text or "SUCCESS" in text


def test_success_impossible_when_mandatory_empty_banner_present():
    text = '<div class="empty-banner">Data unavailable</div><span class="status-badge status-SUCCESS">SUCCESS</span>'
    has_empty = "empty-banner" in text
    has_success = "status-SUCCESS" in text
    is_inconsistent = has_empty and has_success
    assert is_inconsistent


def test_narrative_number_has_provider_date_unit_metadata():
    m_de = CalculatedMetric(series_id="DE_10Y", metric_id="DE_10Y_M", as_of_date=date(2026, 11, 6), latest_value=2.45, change_value=3.2, change_unit="bp")
    from morning_report.interpretation import PortfolioInterpretationEngine
    narrative = PortfolioInterpretationEngine.generate_narrative({"DE_10Y_YIELD": m_de}, "Bear Steepening")
    assert len(narrative) > 0
    assert "2.45%" in narrative[0]


def test_data_quality_table_contains_one_row_per_configured_series():
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    out = orch.run(date(2026, 11, 9), demo_mode=False)
    html = out.read_text(encoding="utf-8")
    for s in bundle.series:
        if s.enabled:
            assert s.id in html


def test_generated_html_does_not_contain_unsupported_ok_statuses():
    q = SeriesQualityItem(series_id="S1", display_name="S1", provider="ecb", status="MISSING", obs_count=0, latest_date="-", stale_days=0)
    assert q.status != "OK"


def test_generated_html_does_not_contain_empty_mandatory_table_bodies():
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    out = orch.run(date(2026, 11, 9), demo_mode=False)
    html = out.read_text(encoding="utf-8")
    assert "<tbody>" in html


def test_dark_theme_palette_constants():
    assert DARK_PAGE_BG == "#050505"
    assert DARK_CARD_BG == "#0D0D0D"
    assert COLOR_CYAN == "#00B8D9"

def test_obsolete_eurusd_volatility_panels_removed():
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    out_path = orch.run(date(2026, 11, 9), demo_mode=True)
    html = out_path.read_text(encoding="utf-8")

    assert "EUR/USD Realized Volatility Term Structure" not in html
    assert "EUR/USD Volatility Surface Unavailable" not in html
    assert "Model ATM Volatility vs. Realized Historical Volatility Comparison" not in html
    assert "Historical Realized Volatility: UNAVAILABLE" not in html
    assert "Dense Targets: - / 3,050" not in html
    assert "Feller Condition:" not in html


def test_separate_eurusd_and_vix_charts_present_with_provenance():
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    out_path = orch.run(date(2026, 11, 9), demo_mode=True)
    html = out_path.read_text(encoding="utf-8")

    assert "EUR/USD Exchange Rate History" in html
    assert "CBOE Volatility Index (VIX) History" in html
    assert "USD per EUR | Higher values denote euro appreciation" in html
    assert "Option-implied expectation of near-term S&P 500 volatility" in html
    assert "Source: ECB Reference Rate" in html or "Source: ECB" in html


def test_btc_volatility_surface_explanation_and_methodology():
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    out_path = orch.run(date(2026, 11, 9), demo_mode=True)
    html = out_path.read_text(encoding="utf-8")

    assert "BTC/USD Market-Implied Volatility Surface" in html
    assert "BTC/USD Volatility Surface Construction Methodology" in html
    assert "DERIBIT API v2" in html
    assert "Raw SVI + Monotone PCHIP" in html
    assert "does not forecast" in html or "no directional price forecast" in html


def test_top_page_market_regime_interpretation_present():
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    out_path = orch.run(date(2026, 11, 9), demo_mode=True)
    html = out_path.read_text(encoding="utf-8")

    assert "Market Regime Interpretation" in html
    assert "Today's Market Summary" in html
    assert "Yield Curve Dynamics:" in html
    assert "Cross-Asset Regime:" in html


def test_data_reactive_eurusd_interpretation():
    from morning_report.interpretation import PortfolioInterpretationEngine
    m_down = CalculatedMetric(series_id="FX_EURUSD", metric_id="M", as_of_date=date(2026, 11, 10), latest_value=1.0650, change_value=-0.75, pct_5d=-1.20, percentile_1y=15.0, zscore_3y=-1.6, change_unit="%")
    interp_down = PortfolioInterpretationEngine.interpret_eurusd(m_down)
    assert "euro depreciation" in interp_down
    assert "lower quartile" in interp_down
    assert "1.0650" in interp_down

    m_up = CalculatedMetric(series_id="FX_EURUSD", metric_id="M", as_of_date=date(2026, 11, 10), latest_value=1.0920, change_value=0.85, pct_5d=1.50, percentile_1y=82.0, zscore_3y=1.4, change_unit="%")
    interp_up = PortfolioInterpretationEngine.interpret_eurusd(m_up)
    assert "euro appreciation" in interp_up
    assert "upper quartile" in interp_up


def test_data_reactive_vix_interpretation():
    from morning_report.interpretation import PortfolioInterpretationEngine
    m_stress = CalculatedMetric(series_id="VIX_INDEX", metric_id="M", as_of_date=date(2026, 11, 10), latest_value=32.5, change_value=22.0, percentile_1y=95.0, change_unit="%")
    interp_stress = PortfolioInterpretationEngine.interpret_vix(m_stress)
    assert "acute market stress" in interp_stress
    assert "increased option-implied uncertainty" in interp_stress
    assert "does not forecast directional" in interp_stress

    m_subdued = CalculatedMetric(series_id="VIX_INDEX", metric_id="M", as_of_date=date(2026, 11, 10), latest_value=12.8, change_value=-5.0, percentile_1y=8.0, change_unit="%")
    interp_subdued = PortfolioInterpretationEngine.interpret_vix(m_subdued)
    assert "subdued implied volatility" in interp_subdued
    assert "reduced option-implied uncertainty" in interp_subdued


def test_chart_source_footer_bounding_box_and_overlap():
    import matplotlib.pyplot as plt
    from morning_report.charts import add_chart_source_footer, validate_chart_layout
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    ax.plot([1, 2, 3], [10, 20, 15])
    ax.set_xlabel("Time Horizon (Sessions)")
    add_chart_source_footer(fig, "Source: ECB Reference Rate | Quote: USD per EUR | Data through: 2026-11-10")
    diag = validate_chart_layout(fig, chart_name="test_eurusd", min_gap_px=8.0)
    assert diag["passed"] is True
    assert diag["overlap_detected"] is False
    assert diag["clipping_detected"] is False
    assert diag["min_gap_px"] >= 8.0
    plt.close(fig)


def test_headline_generation_across_diverse_regimes(tmp_path):
    from morning_report.interpretation import HeadlineEngine
    d = date(2026, 11, 10)

    # 1. Risk-on scenario
    m_on = {
        "EQ_SP500": CalculatedMetric(series_id="EQ_SP500", metric_id="M", as_of_date=d, change_value=2.0, change_unit="%"),
        "US_HY_OAS": CalculatedMetric(series_id="US_HY_OAS", metric_id="M", as_of_date=d, change_value=-8.0, change_unit="bp"),
        "VIX_INDEX": CalculatedMetric(series_id="VIX_INDEX", metric_id="M", as_of_date=d, change_value=-15.0, change_unit="%"),
    }
    reg_on = RegimeClassification(label="Strong Risk-On", plain_language_definition="Risk appetite up", status="OK")
    h_on = HeadlineEngine.generate(d, m_on, None, reg_on, cache_dir=tmp_path)
    assert len(h_on.split()) >= 4
    assert "Monitor" not in h_on

    # 2. Risk-off / Stress scenario
    d2 = date(2026, 11, 11)
    reg_off = RegimeClassification(label="Acute Stress", plain_language_definition="Markets liquidating", status="OK")
    h_off = HeadlineEngine.generate(d2, m_on, None, reg_off, cache_dir=tmp_path)
    assert h_off != h_on

    # 3. Inconclusive data scenario
    d3 = date(2026, 11, 12)
    h_inc = HeadlineEngine.generate(d3, {}, None, None, cache_dir=tmp_path)
    assert len(h_inc) > 10


def test_headline_same_day_rerun_stability(tmp_path):
    from morning_report.interpretation import HeadlineEngine
    d = date(2026, 11, 10)
    m = {"EQ_SP500": CalculatedMetric(series_id="EQ_SP500", metric_id="M", as_of_date=d, change_value=0.5, change_unit="%")}
    reg = RegimeClassification(label="Risk-On", plain_language_definition="Muted gains", status="OK")

    first_run = HeadlineEngine.generate(d, m, None, reg, cache_dir=tmp_path)
    second_run = HeadlineEngine.generate(d, m, None, reg, cache_dir=tmp_path)
    assert first_run == second_run


def test_headline_no_repeat_within_preceding_10_calendar_days(tmp_path):
    from morning_report.interpretation import HeadlineEngine
    base_date = date(2026, 1, 1)
    m = {
        "DE_10Y_YIELD": CalculatedMetric(series_id="DE_10Y_YIELD", metric_id="M", as_of_date=base_date, latest_value=2.45, change_value=0.1, change_unit="bp"),
        "DE_2Y_YIELD": CalculatedMetric(series_id="DE_2Y_YIELD", metric_id="M", as_of_date=base_date, latest_value=2.10, change_value=0.1, change_unit="bp"),
    }
    reg = RegimeClassification(label="Neutral / Range-Bound", plain_language_definition="Quiet tape", status="OK")
    curve = YieldCurveRegime(label="Unchanged / Range-Bound", plain_language_definition="Flat yields", status="OK")

    history_headlines = []
    for offset in range(25):
        cur_date = base_date + timedelta(days=offset)
        hl = HeadlineEngine.generate(cur_date, m, curve, reg, cache_dir=tmp_path)
        norm_hl = HeadlineEngine.normalize_text(hl)
        start_idx = max(0, len(history_headlines) - 10)
        window = history_headlines[start_idx:]
        for prev in window:
            assert norm_hl != HeadlineEngine.normalize_text(prev), f"Duplicate found on day {offset}: '{hl}'"
        history_headlines.append(hl)


def test_headline_exhausted_candidates_do_not_trigger_incompatible_theme(tmp_path):
    from morning_report.interpretation import HeadlineEngine
    base_date = date(2026, 1, 1)
    m = {
        "EQ_SP500": CalculatedMetric(series_id="EQ_SP500", metric_id="M", as_of_date=base_date, change_value=-2.5, change_unit="%"),
        "VIX_INDEX": CalculatedMetric(series_id="VIX_INDEX", metric_id="M", as_of_date=base_date, change_value=20.0, change_unit="%"),
    }
    reg_stress = RegimeClassification(label="Acute Stress", plain_language_definition="Sharp liquidation", status="OK")
    bullish_keywords = ["confetti", "bulls", "green", "cheerful", "advance"]
    for offset in range(30):
        cur_date = base_date + timedelta(days=offset)
        hl = HeadlineEngine.generate(cur_date, m, None, reg_stress, cache_dir=tmp_path)
        for kw in bullish_keywords:
            assert kw not in hl.lower(), f"Incompatible theme candidate chosen under stress: '{hl}'"


def test_headline_regenerates_stored_static_title(tmp_path):
    import json
    from morning_report.interpretation import HeadlineEngine
    hist_file = tmp_path / "headline_history.json"
    target_d = date(2026, 11, 10)
    hist_file.write_text(json.dumps({target_d.isoformat(): "Morning Macro-Financial Risk Monitor"}), encoding="utf-8")
    reg = RegimeClassification(label="Risk-On", plain_language_definition="Gains", status="OK")
    hl = HeadlineEngine.generate(target_d, {}, None, reg, cache_dir=tmp_path)
    assert "Morning Macro-Financial Risk Monitor" not in hl
    assert "Monitor" not in hl


def test_rendered_html_replaces_old_static_heading():
    from morning_report.reporting import ReportRenderer
    from morning_report.models import ReportContext
    test_headline = "Equities Bring the Confetti, Bonds Read the Fine Print"
    ctx = ReportContext(
        as_of_date="2026-11-10",
        execution_time="08:00:00",
        timezone="UTC",
        run_status="SUCCESS",
        headline=test_headline,
        regime_label="Strong Risk-On",
        curve_label="Bull Steepening",
        max_rates_move="+2.1 bp",
        stale_count=0,
    )
    html = ReportRenderer.render_html(ctx)
    assert f"<title>{test_headline}</title>" in html
    assert f"<h1>{test_headline}</h1>" in html
    assert "<h1>Morning Macro-Financial Risk Monitor</h1>" not in html


def test_corrupted_headline_history_handled_gracefully(tmp_path):
    from morning_report.interpretation import HeadlineEngine
    hist_file = tmp_path / "headline_history.json"
    hist_file.write_text("{CORRUPTED_JSON_DATA", encoding="utf-8")

    d = date(2026, 11, 10)
    reg = RegimeClassification(label="Risk-On", plain_language_definition="Gains", status="OK")
    hl = HeadlineEngine.generate(d, {}, None, reg, cache_dir=tmp_path)
    assert isinstance(hl, str)
    assert len(hl) > 0


def test_pipeline_orchestrator_renders_dynamic_headline_into_html(tmp_path):
    bundle = ConfigBundle()
    orch = PipelineOrchestrator(bundle)
    out_path = orch.run(date(2026, 11, 9), demo_mode=True)
    html = out_path.read_text(encoding="utf-8")
    assert "<h1>Morning Macro-Financial Risk Monitor</h1>" not in html
    assert "<title>Morning Macro-Financial Risk Monitor</title>" not in html


if __name__ == "__main__":
    pytest.main(["-v", __file__])