"""Pipeline execution orchestrator."""

from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import pandas as pd
from morning_report.charts import (
    render_credit_history_chart,
    render_cross_asset_zscore_chart,
    render_eurusd_history_chart,
    render_rates_history_chart,
    render_sovereign_yield_curves,
    render_vix_history_chart,
    render_volatility_surface_3d,
    render_volatility_surface_heatmap,
)
from morning_report.market_calendar import MarketCalendarService
from morning_report.config import ConfigBundle
from morning_report.interpretation import PortfolioInterpretationEngine
from morning_report.models import (
    CalculatedMetric, CanonicalSeriesResult, CurvePoint, CurveSnapshot,
    DataQualitySummary, RelativeValueSpread, ReportContext, SeriesQualityItem,
    SovereignRateRow, SurfaceStatus, TableRowItem,
)
from morning_report.providers import (
    DeribitOptionsProvider, ECBDataProvider, FREDDataProvider, RSSNewsProvider, YahooFinanceDataProvider,
)
from morning_report.reporting import ReportRenderer
from morning_report.signals import SignalEngine
from morning_report.transformations import TransformationEngine
from morning_report.validation import DataValidationService
from morning_report.volatility_surface import BTCVolatilitySurfaceEngine

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, config: ConfigBundle) -> None:
        self.config = config
        self.calendar = MarketCalendarService()
        self.validator = DataValidationService(config.thresholds)
        self.signals = SignalEngine(config.thresholds)
        self.ecb = ECBDataProvider()
        self.fred = FREDDataProvider()
        self.yahoo = YahooFinanceDataProvider()
        self.deribit = DeribitOptionsProvider(config.deribit_volatility_surface)
        self.news = RSSNewsProvider(config.news_sources)
        self.output_dir = config.root_dir / config.settings.output_directory
        self.btc_surface_engine = BTCVolatilitySurfaceEngine(config.deribit_volatility_surface)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.ecb.close()
        self.fred.close()
        self.deribit.close()

    def run(self, target_date: date, demo_mode: bool = False) -> Path:
        canonical_results: dict[str, CanonicalSeriesResult] = {}
        histories: dict[str, pd.DataFrame] = {}
        metrics: dict[str, CalculatedMetric] = {}
        start_date = date(2015, 1, 1)

        for s in self.config.series:
            if not s.enabled:
                continue
            res = CanonicalSeriesResult(
                series_id=s.id,
                display_name=s.display_name,
                asset_class=s.asset_class,
                country=s.country,
                currency=s.currency,
                tenor=s.tenor,
                unit=s.unit,
                provider_requested=s.provider,
                fallback_provider=s.secondary_provider,
                requested_report_date=target_date,
                stale_after_days=s.stale_after_days,
                expected_release_lag=s.expected_release_lag,
            )
            df = pd.DataFrame()
            provider_used = None

            if s.provider == "ecb":
                df = self.ecb.fetch_series(s, start_date, target_date)
                provider_used = "ecb"
            elif s.provider == "fred":
                df = self.fred.fetch_series(s, start_date, target_date)
                provider_used = "fred"
            elif s.provider in ["market_data", "licensed", "yfinance"]:
                df = self.yahoo.fetch_series(s, start_date, target_date)
                provider_used = "yfinance"

            # Secondary provider fallback
            if df.empty and s.secondary_provider == "yfinance":
                df = self.yahoo.fetch_series(s, start_date, target_date)
                if not df.empty:
                    provider_used = "yfinance"

            # Demonstration synthetic mode strictly when demo_mode=True
            if df.empty and demo_mode:
                dates = [target_date - pd.Timedelta(days=i) for i in range(10, -1, -1)]
                base_val = 2.45 if s.asset_class == "rates" else (480.0 if s.asset_class == "equities" else 1.08)
                df = pd.DataFrame([{"observation_date": d, "value": base_val + (i * 0.01)} for i, d in enumerate(dates)])
                provider_used = "synthetic_demo"
                res.synthetic = True

            if not df.empty:
                clean = df.dropna(subset=["observation_date", "value"]).copy()
                clean["observation_date"] = pd.to_datetime(clean["observation_date"]).dt.date
                clean = clean[clean["observation_date"] <= target_date]
                clean = clean.sort_values("observation_date").drop_duplicates(subset=["observation_date"])
                res.history = clean
                res.raw_observation_count = len(df)
                res.valid_observation_count = len(clean)
                if not clean.empty:
                    res.first_observation_date = clean.iloc[0]["observation_date"]
                    res.latest_observation_date = clean.iloc[-1]["observation_date"]
                    res.effective_observation_date = clean.iloc[-1]["observation_date"]
                    res.age_calendar_days = (target_date - res.effective_observation_date).days
                    res.stale = res.age_calendar_days > s.stale_after_days
                    res.provider_used = provider_used
                    res.retrieval_status = "RETRIEVED"
                    histories[s.id] = clean

                    v_res = self.validator.validate_series_data(s, clean, target_date)
                    res.validation_status = "PASSED" if v_res.passed else "FAILED"
                    res.warnings.extend(v_res.warnings)
                    res.errors.extend(v_res.errors)
                    metrics[s.id] = TransformationEngine.calculate_metrics(s, clean, target_date)
                else:
                    res.retrieval_status = "EMPTY_PAYLOAD"
                    res.validation_status = "EMPTY"
                    metrics[s.id] = TransformationEngine.calculate_metrics(s, pd.DataFrame(), target_date)
            else:
                status = "MISSING_API_KEY" if s.provider == "fred" and not self.fred.api_key else "MISSING"
                res.retrieval_status = status
                res.validation_status = "MISSING"
                res.errors.append(f"Failed retrieval from {s.provider}")
                metrics[s.id] = TransformationEngine.calculate_metrics(s, pd.DataFrame(), target_date)

            canonical_results[s.id] = res
            logger.info(
                "%s | %s | obs: %d | first: %s | latest: %s | eff: %s | %s",
                s.id, res.provider_used or res.provider_requested,
                res.valid_observation_count, res.first_observation_date,
                res.latest_observation_date, res.effective_observation_date,
                res.retrieval_status,
            )

        # Build sovereign curve snapshots
        curve_snapshots: list[CurveSnapshot] = []
        for c_group, c_name, ccy in [("DE_BUND", "Euro Area AAA Benchmark", "EUR"), ("US_TREASURY", "US Treasury", "USD"), ("FI_GOVT", "Finland Sovereign", "EUR")]:
            rate_defs = [s for s in self.config.series if s.curve_group == c_group and s.tenor_years is not None]
            points: list[CurvePoint] = []
            for s_def in rate_defs:
                m = metrics.get(s_def.id)
                if m and m.latest_value is not None and m.as_of_date is not None:
                    points.append(CurvePoint(
                        tenor=s_def.tenor or f"{s_def.tenor_years}Y",
                        tenor_years=s_def.tenor_years,
                        yield_pct=m.latest_value,
                        observation_date=m.as_of_date,
                        instrument_id=s_def.id,
                    ))
            if points:
                points.sort(key=lambda p: p.tenor_years)
                passed, w_list, e_list = self.validator.validate_curve(points)
                curve_snapshots.append(CurveSnapshot(
                    curve_id=c_group,
                    display_name=c_name,
                    country=c_group.split("_")[0],
                    currency=ccy,
                    as_of_date=points[-1].observation_date,
                    points=points,
                    is_complete=passed,
                    warnings=w_list + e_list,
                ))

        # Sovereign Rates Table
        sovereign_rates_table: list[SovereignRateRow] = []
        for snap in curve_snapshots:
            c_code = snap.country
            p_map = {p.tenor: p for p in snap.points}
            m10 = metrics.get(f"{c_code}_10Y_YIELD") or next((m for sid, m in metrics.items() if sid.startswith(c_code) and "10Y" in sid), None)
            p2 = p_map.get("2Y")
            p10 = p_map.get("10Y")
            p5 = p_map.get("5Y")
            p30 = p_map.get("30Y")
            s_2s10s = f"{(p10.yield_pct - p2.yield_pct) * 100.0:+.1f} bp" if (p10 and p2) else "-"
            s_5s30s = f"{(p30.yield_pct - p5.yield_pct) * 100.0:+.1f} bp" if (p30 and p5) else "-"
            sovereign_rates_table.append(SovereignRateRow(
                country_code=c_code,
                country_name=snap.display_name,
                currency=snap.currency,
                yield_2y=f"{p2.yield_pct:.2f}%" if p2 else "N/A",
                yield_5y=f"{p5.yield_pct:.2f}%" if p5 else "N/A",
                yield_10y=f"{p10.yield_pct:.2f}%" if p10 else "N/A",
                yield_30y=f"{p30.yield_pct:.2f}%" if p30 else "N/A",
                spread_2s10s=s_2s10s,
                spread_5s30s=s_5s30s,
                chg_10y_1d=f"{m10.change_value:+.1f} bp" if m10 and m10.change_value is not None else "-",
                chg_10y_1d_val=m10.change_value if m10 else None,
                chg_10y_5d=f"{m10.pct_5d:+.1f} bp" if m10 and m10.pct_5d is not None else "-",
                chg_10y_1m=f"{m10.pct_1m:+.1f} bp" if m10 and m10.pct_1m is not None else "-",
                zscore_10y_3y=f"{m10.zscore_3y:+.1f}σ" if m10 and m10.zscore_3y is not None else "-",
                as_of=snap.as_of_date.isoformat(),
                data_age_days=(target_date - snap.as_of_date).days,
                provider="ECB" if c_code in ["DE", "FI"] else "FRED",
                status="OK" if snap.is_complete else "PARTIAL",
            ))

        # Relative Value Spreads
        relative_value_spreads: list[RelativeValueSpread] = []
        de10 = metrics.get("DE_10Y_YIELD")
        fi10 = metrics.get("FI_10Y_YIELD")
        if de10 and fi10 and de10.latest_value is not None and fi10.latest_value is not None:
            diff_d = abs((fi10.as_of_date - de10.as_of_date).days) if (fi10.as_of_date and de10.as_of_date) else 0
            spread_val = (fi10.latest_value - de10.latest_value) * 100.0
            q_status = "OK" if diff_d <= 4 and fi10.quality_status != "missing" else "WARNING"
            relative_value_spreads.append(RelativeValueSpread(
                label="Finland 10Y vs German Benchmark 10Y",
                spread_bp=f"{spread_val:+.1f} bp",
                spread_val=spread_val,
                chg_1d=f"{(fi10.change_value or 0.0) - (de10.change_value or 0.0):+.1f} bp" if fi10.change_value is not None and de10.change_value is not None else "-",
                long_date=str(fi10.as_of_date),
                benchmark_date=str(de10.as_of_date),
                date_diff_days=diff_d,
                provider="ECB SDMX",
                quality_status=q_status,
                commentary="Nordic sovereign risk premium over Euro area AAA benchmark.",
            ))

        de2 = metrics.get("DE_2Y_YIELD")
        yield_curve_regime = self.signals.classify_yield_curve(
            de2.change_value if de2 else None,
            de10.change_value if de10 else None,
            histories=histories,
            target_date=target_date,
        )
        curve_label = yield_curve_regime.label
        cross_asset_regime = self.signals.classify_cross_asset_regime(metrics)
        regime_label = cross_asset_regime.label

        # Render Sovereign Curve Chart from validated snapshots
        valid_curve_snaps = [s for s in curve_snapshots if len(s.points) >= 4]
        sovereign_curve_chart = render_sovereign_yield_curves(valid_curve_snaps) if valid_curve_snaps else None

        # Benchmark Yield Rolling 3Y History Chart
        rates_history_chart = render_rates_history_chart(histories, self.config.series) if any(k in histories for k in ["DE_2Y_YIELD", "DE_10Y_YIELD", "US_10Y_YIELD"]) else None

        # Credit Spread OAS History Chart
        credit_history_chart = render_credit_history_chart(histories) if "US_HY_OAS" in histories or "US_IG_OAS" in histories else None

        # Independent EUR/USD History Chart with Provenance
        eurusd_as_of = str(metrics.get("FX_EURUSD").as_of_date) if (metrics.get("FX_EURUSD") and metrics.get("FX_EURUSD").as_of_date) else str(target_date)
        eurusd_res = canonical_results.get("FX_EURUSD")
        eurusd_prov = eurusd_res.provider_used if eurusd_res else "ECB"
        if eurusd_prov == "yfinance":
            eurusd_src = f"Source: Yahoo Finance (EURUSD=X, Fallback) | Quote: USD per EUR | Trans: Spot Level"
        else:
            eurusd_src = f"Source: ECB Reference Rate (EXR.D.USD.EUR.SP00.A) | Quote: USD per EUR | Trans: Spot Level"
        eurusd_chart = render_eurusd_history_chart(histories, source_footer=eurusd_src, data_through=eurusd_as_of) if "FX_EURUSD" in histories and not histories["FX_EURUSD"].empty else None

        # Independent VIX History Chart with Provenance
        vix_as_of = str(metrics.get("VIX_INDEX").as_of_date) if (metrics.get("VIX_INDEX") and metrics.get("VIX_INDEX").as_of_date) else str(target_date)
        vix_res = canonical_results.get("VIX_INDEX")
        vix_prov = vix_res.provider_used if vix_res else "FRED"
        if vix_prov == "yfinance":
            vix_src = f"Source delivery: Yahoo Finance (^VIX, Fallback) | Methodology owner: Cboe | Trans: Close Level"
        else:
            vix_src = f"Source delivery: FRED VIXCLS | Methodology owner: Cboe | Expected 30D S&P 500 Implied Volatility"
        vix_chart = render_vix_history_chart(histories, source_footer=vix_src, data_through=vix_as_of) if "VIX_INDEX" in histories and not histories["VIX_INDEX"].empty else None

        # BTC/USD Market-Implied Volatility Surface via Deribit Public API
        now_utc = datetime.now(timezone.utc)
        deribit_observations, deribit_meta = self.deribit.fetch_btc_option_chain(now_utc)
        btc_surface_res = self.btc_surface_engine.build_btc_surface(deribit_observations, now_utc)

        btc_heatmap_chart = None
        btc_3d_chart = None
        if btc_surface_res.status in [SurfaceStatus.AVAILABLE, SurfaceStatus.AVAILABLE_WITH_WARNINGS]:
            btc_heatmap_chart = render_volatility_surface_heatmap(btc_surface_res)
            btc_3d_chart = render_volatility_surface_3d(btc_surface_res)

        # Cross-Asset Z-Score Chart (strict acceptance criteria: non-null, genuine history)
        z_labels, z_vals = [], []
        for sid, label in [("EQ_SP500", "S&P 500"), ("EQ_EUROPE_600", "Eq Europe"), ("US_HY_OAS", "US HY OAS"), ("VIX_INDEX", "VIX"), ("FX_EURUSD", "EUR/USD")]:
            if sid in metrics and metrics[sid].zscore_3y is not None and not canonical_results[sid].synthetic:
                z_labels.append(label)
                z_vals.append(metrics[sid].zscore_3y)
        zscore_chart = render_cross_asset_zscore_chart(z_labels, z_vals) if z_labels else None

        # Market Table Rows
        market_table: list[TableRowItem] = []
        market_sids = ["EQ_SP500", "EQ_EUROPE_600", "US_HY_OAS", "US_IG_OAS", "VIX_INDEX", "FX_EURUSD", "ESTR_RATE"]
        for sid in market_sids:
            s_def = next((s for s in self.config.series if s.id == sid), None)
            if not s_def:
                continue
            m = metrics.get(sid)
            c_res = canonical_results.get(sid)
            lvl_str = f"{m.latest_value:.2f}" if (m and m.latest_value is not None) else "N/A"
            unit_str = s_def.unit
            if s_def.asset_class == "credit" and lvl_str != "N/A":
                lvl_str = f"{m.latest_value:.0f} bp"
                unit_str = "bp"
            market_table.append(TableRowItem(
                name=s_def.display_name,
                level=lvl_str,
                unit=unit_str,
                chg=f"{m.change_value:+.2f} {m.change_unit}" if (m and m.change_value is not None) else "-",
                chg_val=m.change_value if m else None,
                chg_5d=f"{m.pct_5d:+.1f}{m.change_unit}" if (m and m.pct_5d is not None) else "-",
                chg_1m=f"{m.pct_1m:+.1f}{m.change_unit}" if (m and m.pct_1m is not None) else "-",
                chg_ytd=f"{m.pct_ytd:+.1f}%" if (m and m.pct_ytd is not None) else "-",
                pctile=f"{m.percentile_1y:.0f}%" if (m and m.percentile_1y is not None) else "-",
                zscore=f"{m.zscore_3y:+.1f}σ" if (m and m.zscore_3y is not None) else "-",
                as_of=str(m.as_of_date) if (m and m.as_of_date) else "-",
                status="OK" if (m and m.quality_status == "ok") else (m.quality_status.upper() if m else "MISSING"),
                is_proxy=s_def.is_proxy,
                country=s_def.country or s_def.asset_class.upper(),
                provider=c_res.provider_used or s_def.provider if c_res else s_def.provider,
            ))

        # Data Quality & Governance Summary
        quality_items: list[SeriesQualityItem] = []
        for s_def in self.config.series:
            c_res = canonical_results.get(s_def.id)
            obs_cnt = c_res.valid_observation_count if c_res else 0
            lat_d = str(c_res.latest_observation_date or "-") if c_res else "-"
            eff_d = str(c_res.effective_observation_date or "-") if c_res else "-"
            stale_d = c_res.age_calendar_days if c_res else 0
            st = "OK"
            if not c_res or c_res.retrieval_status in ["MISSING", "MISSING_API_KEY"]:
                st = "MISSING"
            elif c_res.stale:
                st = "STALE"
            elif obs_cnt < 252:
                st = "INSUFFICIENT_HISTORY"
            err_detail = c_res.errors[0] if (c_res and c_res.errors) else (c_res.retrieval_error if c_res else None)
            quality_items.append(SeriesQualityItem(
                series_id=s_def.id,
                display_name=s_def.display_name,
                provider=(c_res.provider_used or s_def.provider) if c_res else s_def.provider,
                status=st,
                obs_count=obs_cnt,
                latest_date=lat_d,
                effective_date=eff_d,
                stale_days=stale_d,
                is_demo=c_res.synthetic if c_res else False,
                retrieval_status=c_res.retrieval_status if c_res else "FAILED",
                error_detail=err_detail,
                warnings=c_res.warnings if c_res else [],
                errors=c_res.errors if c_res else [],
            ))

        # Explicit Data Quality Governance for Deribit BTC Options and Calibration
        raw_inst_cnt = deribit_meta.get("raw_instruments_count", 0)
        valid_obs_cnt = btc_surface_res.valid_observation_count
        cache_st = deribit_meta.get("cache_status", "MISS")
        ret_time_str = deribit_meta.get("retrieval_timestamp", now_utc.isoformat())[:16]

        quality_items.append(SeriesQualityItem(
            series_id="DERIBIT_BTC_OPTION_INSTRUMENTS",
            display_name="Deribit BTC Active Option Instruments Discovery",
            provider="DERIBIT",
            status="OK" if raw_inst_cnt > 0 else "MISSING",
            obs_count=raw_inst_cnt,
            latest_date=ret_time_str,
            effective_date=ret_time_str,
            stale_days=0,
            cache_status=cache_st,
            retrieval_status=self.deribit.last_status,
            error_detail=self.deribit.last_error if raw_inst_cnt == 0 else None,
        ))
        quality_items.append(SeriesQualityItem(
            series_id="DERIBIT_BTC_OPTION_QUOTES",
            display_name="Deribit BTC Market Quotes & Mark IV Snapshots",
            provider="DERIBIT",
            status="OK" if valid_obs_cnt >= 80 else ("WARNING" if valid_obs_cnt > 0 else "MISSING"),
            obs_count=valid_obs_cnt,
            latest_date=ret_time_str,
            effective_date=ret_time_str,
            stale_days=0,
            cache_status=cache_st,
            retrieval_status=self.deribit.last_status,
            error_detail=f"Retained {btc_surface_res.retained_quote_count} quotes across {len(btc_surface_res.retained_expiries)} expiries.",
        ))
        quality_items.append(SeriesQualityItem(
            series_id="BTC_VOL_SURFACE_CALIBRATION",
            display_name="BTC/USD Market-Implied SVI Surface Calibration",
            provider="DERIBIT",
            status="OK" if btc_surface_res.status in [SurfaceStatus.AVAILABLE, SurfaceStatus.AVAILABLE_WITH_WARNINGS] else "FAILED",
            obs_count=btc_surface_res.retained_quote_count,
            latest_date=ret_time_str,
            effective_date=ret_time_str,
            stale_days=0,
            retrieval_status=btc_surface_res.status.value,
            error_detail=f"Weighted RMSE: {btc_surface_res.weighted_rmse}%, Dense coverage: {btc_surface_res.dense_grid_coverage_pct}%",
            warnings=btc_surface_res.warnings,
        ))

        q_summary = DataQualitySummary(
            total_series=len(quality_items),
            valid_series=sum(1 for q in quality_items if q.status == "OK"),
            stale_series=sum(1 for q in quality_items if q.status == "STALE"),
            missing_series=sum(1 for q in quality_items if q.status == "MISSING"),
            insufficient_history=sum(1 for q in quality_items if q.status == "INSUFFICIENT_HISTORY"),
            missing_api_keys=sum(1 for q in quality_items if "MISSING_API_KEY" in q.retrieval_status),
            items=quality_items,
        )

        # Central Status Determination
        has_mandatory_curve = len(valid_curve_snaps) > 0
        has_sovereign_table = len(sovereign_rates_table) > 0
        has_rates_history = rates_history_chart is not None
        has_credit_history = credit_history_chart is not None
        has_eurusd = eurusd_chart is not None
        has_vix = vix_chart is not None
        has_market_table = len(market_table) > 0

        if has_mandatory_curve and has_sovereign_table and has_rates_history and has_credit_history and (has_eurusd or has_vix) and has_market_table:
            run_status = "SUCCESS_WITH_WARNINGS" if q_summary.stale_series > 0 or q_summary.missing_series > 0 else "SUCCESS"
        elif has_sovereign_table or has_market_table:
            run_status = "PARTIAL"
        else:
            run_status = "FAILED"

        # Release lag assessment
        max_lag = max([c.age_calendar_days for c in canonical_results.values() if c.valid_observation_count > 0] or [0])
        lag_rec = "No lag required; feeds current." if max_lag == 0 else (
            "1-business-day publication lag recommended for late-reporting feeds." if max_lag <= 2 else
            f"Provider-specific publication lag or feed offline (actual lag: {max_lag} days)."
        )

        today_summary = PortfolioInterpretationEngine.generate_today_market_summary(metrics, yield_curve_regime, cross_asset_regime, btc_surface_res)
        narrative = PortfolioInterpretationEngine.generate_narrative(metrics, curve_label)
        sovereign_interp = PortfolioInterpretationEngine.interpret_sovereign_curves(yield_curve_regime)
        rates_interp = f"German 10Y yield trades at {de10.latest_value:.2f}% ({de10.change_value:+.1f} bp 1D). Historical sovereign benchmarks reflect ECB deposit facility expectations and euro-area term premium." if de10 and de10.latest_value is not None else "Historical rate observations pending feed update."
        credit_interp = PortfolioInterpretationEngine.interpret_credit(metrics.get("US_HY_OAS"))
        eurusd_interp = PortfolioInterpretationEngine.interpret_eurusd(metrics.get("FX_EURUSD"))
        vix_interp = PortfolioInterpretationEngine.interpret_vix(metrics.get("VIX_INDEX"))
        zscore_interp = "Normalized 3-year z-scores highlight cross-asset deviations beyond ±1.5σ. Current moves reflect localized sector repositioning rather than systemic multi-sigma disruption."
        btc_interp = PortfolioInterpretationEngine.interpret_btc_surface(btc_surface_res)
        news = self.news.fetch_news()

        headline = PortfolioInterpretationEngine.generate_headline(
            target_date=target_date,
            metrics=metrics,
            curve_regime=yield_curve_regime,
            cross_asset_regime=cross_asset_regime,
            cache_dir=self.config.root_dir / self.config.settings.cache_dir,
        )

        context = ReportContext(
            as_of_date=target_date.isoformat(),
            execution_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
            timezone=self.config.settings.timezone,
            run_status=run_status,
            headline=headline,
            regime_label=regime_label,
            curve_label=curve_label,
            today_market_summary=today_summary,
            yield_curve_regime=yield_curve_regime,
            cross_asset_regime_detail=cross_asset_regime,
            max_rates_move=f"{de10.change_value:+.1f} bp" if de10 and de10.change_value else "0.0 bp",
            stale_count=q_summary.stale_series,
            missing_count=q_summary.missing_series,
            missing_api_keys=q_summary.missing_api_keys,
            insufficient_history_count=q_summary.insufficient_history,
            release_lag_summary=f"Max lag: {max_lag}d",
            lag_recommendation=lag_rec,
            sovereign_rates_table=sovereign_rates_table,
            relative_value_spreads=relative_value_spreads,
            market_table=market_table,
            sovereign_curve_chart_uri=sovereign_curve_chart,
            rates_history_chart_uri=rates_history_chart,
            credit_history_chart_uri=credit_history_chart,
            eurusd_chart_uri=eurusd_chart,
            vix_chart_uri=vix_chart,
            sovereign_curve_interpretation=sovereign_interp,
            rates_history_interpretation=rates_interp,
            credit_history_interpretation=credit_interp,
            eurusd_interpretation=eurusd_interp,
            vix_interpretation=vix_interp,
            zscore_interpretation=zscore_interp,
            btc_surface_interpretation=btc_interp,
            vol_surface_chart_uri=btc_heatmap_chart,
            vol_surface_3d_chart_uri=btc_3d_chart,
            btc_volatility_surface=btc_surface_res,
            zscore_chart_uri=zscore_chart,
            quality_summary=q_summary,
            narrative_bullets=narrative,
            news_items=[n.model_dump() for n in news],
            demo_mode=demo_mode,
        )

        html_path = self.output_dir / f"{target_date.isoformat()}_morning_macro_report.html"
        pdf_path = self.output_dir / f"{target_date.isoformat()}_morning_macro_report.pdf"
        html_content = ReportRenderer.render_html(context)
        html_path.write_text(html_content, encoding="utf-8")
        ReportRenderer.render_pdf(html_content, pdf_path)
        print(f"Dynamic headline: {headline}")
        print(f"Report compiled to: {html_path.resolve()}")
        return html_path


if __name__ == "__main__":
    bundle = ConfigBundle()
    orchestrator = PipelineOrchestrator(bundle)
    target_dt = orchestrator.calendar.get_previous_business_day(date.today())
    output_report = orchestrator.run(target_dt)
    print(f"Morning risk report compiled successfully: {output_report}")