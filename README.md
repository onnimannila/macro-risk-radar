# Morning Macro-Financial Risk Monitor

Automated macro-financial risk monitor for pension fund investment risk management designed for institutional balance-sheet decision support.

## Features

- **BTC/USD Market-Implied Volatility Surface**: Real-time option-market calibration to live Deribit public options data across multiple expiries and strikes via Quasi-Explicit SVI with monotone PCHIP total-variance term structure interpolation.
- **Standalone Institutional EUR/USD & VIX Charts**: Independent, high-contrast time-series charts with moving averages, percentile indicators, and data-reactive commentary.
- **Multi-Sovereign Term Structure Curves**: Full yield curves across German Bunds (3M to 30Y) and US Treasuries (3M to 30Y) with numeric maturity ordering.
- **Top-of-Page Market Regime Synthesis**: Deterministic, multi-horizon yield curve (1D, 5D, 1M, 3M) and cross-asset regime classification with explicit component decomposition.
- **Bloomberg-Terminal Dark Theme**: High-contrast dark palette (`#050505` page, `#0D0D0D` panels, `#F2F2F2` primary headings, `#C5CDD8` subtitles) compliant with WCAG 2.1 contrast standards (>7:1).
- **Strict Non-Negotiable Data Policy**: Real-world data failures are logged as `MISSING` or `ERROR` in the governance panel. No fabricated market observations in production.
- **Statistical Rigor**: 1Y percentile calculation over 252 trading observations; 3Y rolling z-scores over up to 756 trading moves; 60D annualized volatility; multi-tenor Max Rates Move.
- **Data Provenance Footers**: Every graph displays authoritative data provider, observation through-date, retrieval timestamp, and transformation methodology.

## Architecture

- `cli.py`: Typer CLI entrypoint (`validate-config`, `discover-ecb`, `provider-status`, `run`).
- `pipeline.py`: Orchestrates retrieval, validation, transformations, chart generation, and rendering.
- `providers.py`: Connects to ECB SDMX REST API, St. Louis Fed FRED API, Deribit public options API, and RSS press feeds.
- `volatility_surface.py`: Deribit BTC options normalization, OTM pairing, SVI multi-start calibration, and dense grid evaluation.
- `transformations.py`: Computes 1D, 5D, 1M, YTD changes, 60D volatility, 1Y percentiles, and 3Y z-scores.
- `validation.py`: Verifies series plausibility thresholds, duplicate dates, and sovereign curve structures.
- `charts.py`: Produces Matplotlib visualizations (sovereign term structures, yield histories, credit bands, and z-score spreads).
- `reporting.py`: Jinja2 template and WeasyPrint PDF renderer.

## Environment Variables

```bash
export FRED_API_KEY="your_stlouisfed_fred_api_key"
```

## Installation

```bash
pip install typer pydantic pyyaml pandas numpy matplotlib httpx feedparser jinja2 weasyprint pandas_market_calendars pytest
```

## Execution Commands

### 1. Validate Configuration
```bash
python -m morning_report.cli validate-config
```

### 2. Check Provider Connectivity
```bash
python -m morning_report.cli provider-status
```

### 3. Run Production Pipeline (Default)
```bash
python -m morning_report.cli run --date latest
```

### 4. Run Demonstration Pipeline (Synthetic Fallback Mode)
```bash
python -m morning_report.cli run --date latest --demo-data
```

## Unit Tests

```bash
pytest -v test.py
```

## Data Quality & Status Classification

- `SUCCESS`: All mandatory series retrieved and verified without anomalies.
- `SUCCESS_WITH_WARNINGS`: Report compiled with stale or incomplete non-critical feeds.
- `PARTIAL`: One or more major data series failed or unconfigured.
- `FAILED`: Mandatory rendering or calendar resolution failure.

Output reports are saved to:
- `output/YYYY-MM-DD_morning_macro_report.html`
- `output/YYYY-MM-DD_morning_macro_report.pdf`