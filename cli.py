"""Command line interface entrypoint."""

from datetime import date, datetime, timezone
from pathlib import Path
import sys
import typer
import json
import webbrowser
import base64

from morning_report.market_calendar import MarketCalendarService
from morning_report.config import ConfigBundle
from morning_report.pipeline import PipelineOrchestrator
from morning_report.providers import DeribitOptionsProvider, ECBDataProvider, FREDDataProvider

app = typer.Typer(help="Morning Macro-Financial Risk Monitor CLI")


@app.command()
def diagnose_runtime() -> None:
    """Inspects resolved execution paths, package roots, and scans for obsolete report sections."""
    import morning_report
    import morning_report.pipeline as p_mod
    import morning_report.reporting as r_mod
    import morning_report.charts as c_mod

    cfg = ConfigBundle()
    typer.echo("=== RUNTIME ENVIRONMENT DIAGNOSTICS ===")
    typer.echo(f"Current Working Directory: {Path.cwd()}")
    typer.echo(f"Python Executable: {sys.executable}")
    typer.echo(f"morning_report.__file__: {morning_report.__file__}")
    typer.echo(f"Imported pipeline.py: {p_mod.__file__}")
    typer.echo(f"Imported reporting.py: {r_mod.__file__}")
    typer.echo(f"Imported charts.py: {c_mod.__file__}")
    typer.echo(f"Loaded Template Source: {r_mod.__file__} (HTML_TEMPLATE)")
    settings_file = cfg.root_dir / "config" / "settings.yaml" if (cfg.root_dir / "config" / "settings.yaml").exists() else cfg.root_dir / "settings.yaml"
    typer.echo(f"Loaded Settings Path: {settings_file}")
    series_file = cfg.root_dir / "config" / "series.yaml" if (cfg.root_dir / "config" / "series.yaml").exists() else cfg.root_dir / "series.yaml"
    typer.echo(f"Loaded Series Config Path: {series_file}")
    output_dir = cfg.root_dir / cfg.settings.output_directory
    typer.echo(f"Output Directory Path: {output_dir.resolve()}")
    typer.echo(f"Report Generation Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")

    obsolete_strings = [
        "Foreign Exchange (EUR/USD) & Volatility (VIX) History",
        "Data unavailable: FX / VIX time series failed",
        "EUR/USD Realized Volatility Term Structure",
        "Historical Realized Volatility",
        "EUR/USD Volatility Surface Unavailable",
        "Model ATM Volatility vs. Realized Historical Volatility Comparison",
        "Dense Targets",
        "Feller Condition",
    ]
    typer.echo("\n=== OBSOLETE SECTION AUDIT ACROSS PRODUCTION CODEBASE ===")
    root_to_scan = Path(__file__).resolve().parent
    found_any = False
    for p in root_to_scan.rglob("*.py"):
        if ".venv" in str(p) or "__pycache__" in str(p) or "test.py" in str(p):
            continue
        content = p.read_text(encoding="utf-8")
        for target in obsolete_strings:
            if target in content:
                found_any = True
                typer.echo(f"[WARN] Found '{target}' in {p.relative_to(root_to_scan)}")
    if not found_any:
        typer.echo("Audit passed: 0 obsolete strings found in production codebase.")


@app.command()
def validate_config() -> None:
    cfg = ConfigBundle()
    typer.echo(f"Configuration valid. Loaded {len(cfg.series)} series definitions.")


@app.command()
def discover_ecb() -> None:
    healthy = ECBDataProvider().health_check()
    typer.echo(f"ECB SDMX REST API: {'ONLINE' if healthy else 'OFFLINE/UNREACHABLE'}")


@app.command()
def run(
    target_date: str = typer.Option("latest", "--date", "-d", help="YYYY-MM-DD or 'latest'"),
    demo_data: bool = typer.Option(False, "--demo-data", help="Enable synthetic fallback data for demonstration/offline review."),
    open_report: bool = typer.Option(False, "--open", help="Open generated HTML report in the default browser.")
) -> None:
    cfg = ConfigBundle()
    cal = MarketCalendarService()
    resolved_date = cal.get_previous_business_day(date.today()) if target_date.lower() == "latest" else date.fromisoformat(target_date)
    typer.echo(f"Executing report pipeline for {resolved_date} (demo_data={demo_data})...")
    html_out = PipelineOrchestrator(cfg).run(resolved_date, demo_mode=demo_data)
    typer.echo(f"Report successfully compiled to: {html_out}")
    if open_report and html_out.exists():
        webbrowser.open(html_out.resolve().as_uri())


@app.command()
def provider_status() -> None:
    typer.echo(f"ECB Data Portal: {'ONLINE' if ECBDataProvider().health_check() else 'OFFLINE'}")
    typer.echo(f"FRED API: {'CONFIGURED' if FREDDataProvider().health_check() else 'MISSING_API_KEY'}")
    typer.echo(f"Deribit Public API: {'ONLINE' if DeribitOptionsProvider().health_check() else 'OFFLINE'}")


@app.command()
def diagnose_btc_vol_surface(
    target_date: str = typer.Option("latest", "--date", "-d", help="YYYY-MM-DD or 'latest'"),
    output_dir: str = typer.Option("diagnostics", "--output-dir", help="Directory for diagnostic files"),
    force_refresh: bool = typer.Option(False, "--force-refresh", help="Bypass local disk cache"),
    currency: str = typer.Option("BTC", "--currency", help="Cryptocurrency underlying"),
) -> None:
    """Executes high-resolution numerical volatility surface diagnostic suite on genuine Deribit option data."""
    import pandas as pd
    from morning_report.charts import render_volatility_surface_heatmap, render_volatility_surface_3d
    from morning_report.volatility_surface import BTCVolatilitySurfaceEngine

    cfg = ConfigBundle()
    cfg.deribit_volatility_surface.currency = currency
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Connecting to Deribit production API v2 for {currency} options (force_refresh={force_refresh})...")
    provider = DeribitOptionsProvider(cfg.deribit_volatility_surface)
    if force_refresh:
        for old in Path(".cache").glob("deribit_btc_chain_*.json"):
            try:
                old.unlink()
            except OSError:
                pass

    now_utc = datetime.now(timezone.utc)
    observations, meta = provider.fetch_btc_option_chain(now_utc)
    typer.echo(f"Retrieved {len(observations)} raw instrument quotes from Deribit (Cache: {meta.get('cache_status')}).")

    engine = BTCVolatilitySurfaceEngine(cfg.deribit_volatility_surface)
    res = engine.build_btc_surface(observations, now_utc)

    # 1. deribit_btc_normalized_quotes.csv
    norm_records = [o.model_dump() for o in observations if o.quality_status == "VALID"]
    pd.DataFrame(norm_records).to_csv(out_path / "deribit_btc_normalized_quotes.csv", index=False)

    # 2. deribit_btc_excluded_quotes.csv
    excl_records = [o.model_dump() for o in observations if o.quality_status != "VALID"]
    pd.DataFrame(excl_records).to_csv(out_path / "deribit_btc_excluded_quotes.csv", index=False)

    # 3. deribit_btc_svi_parameters.csv & deribit_btc_expiry_summary.csv
    p_rows = []
    for exp_k, params in res.calibration_parameters.items():
        p_rows.append({"expiry": exp_k, **{k: v for k, v in params.items() if k != "residuals"}, "rmse_pct": res.calibration_rmse_by_expiry.get(exp_k)})
    pd.DataFrame(p_rows).to_csv(out_path / "deribit_btc_svi_parameters.csv", index=False)

    # 4. deribit_btc_surface_grid.csv
    grid_rows = []
    for i, t_val in enumerate(res.dense_maturity_grid):
        for j, k_val in enumerate(res.dense_moneyness_grid):
            grid_rows.append({
                "maturity_years": t_val,
                "log_forward_moneyness": k_val,
                "forward_moneyness": res.dense_forward_moneyness_grid[j],
                "implied_volatility_pct": (res.dense_vol_matrix[i][j] * 100.0) if res.dense_vol_matrix[i][j] is not None else None,
            })
    pd.DataFrame(grid_rows).to_csv(out_path / "deribit_btc_surface_grid.csv", index=False)

    # 5. Visual renders
    heatmap_b64 = render_volatility_surface_heatmap(res)
    if "," in heatmap_b64:
        (out_path / "deribit_btc_surface_heatmap.png").write_bytes(base64.b64decode(heatmap_b64.split(",")[1]))
    surface_3d_b64 = render_volatility_surface_3d(res)
    if "," in surface_3d_b64:
        (out_path / "deribit_btc_surface_3d.png").write_bytes(base64.b64decode(surface_3d_b64.split(",")[1]))

    # 6. Quality summary json
    (out_path / "deribit_btc_surface_quality.json").write_text(json.dumps({
        "status": res.status.value,
        "retained_quotes": res.retained_quote_count,
        "retained_expiries": res.retained_expiries,
        "weighted_rmse": res.weighted_rmse,
        "coverage_pct": res.dense_grid_coverage_pct,
        "arbitrage_diagnostics": res.arbitrage_diagnostics,
    }, indent=2))
    typer.echo(f"Diagnostics successfully exported to {out_path.resolve()} (Status: {res.status.value}, Weighted RMSE: {res.weighted_rmse}%).")


if __name__ == "__main__":
    app()