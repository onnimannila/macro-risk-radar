"""Bloomberg-inspired high-contrast dark terminal report rendering with data provenance and interpretation."""

from pathlib import Path
import jinja2
from morning_report.models import ReportContext

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ headline if headline else "Morning Macro-Financial Risk Monitor" }}</title>
<style>
  :root {
    --page-bg: #050505;
    --card-bg: #0D0D0D;
    --card-alt: #171717;
    --text-primary: #F2F2F2;      /* WCAG AAA compliant text on #0D0D0D */
    --text-secondary: #AEB8C4;    /* WCAG AA compliant text on #0D0D0D */
    --subtitle: #C5CDD8;          /* High-contrast subtitle */
    --border-color: #293241;
    --gridlines: #33404D;
    --positive: #00E676;
    --negative: #FF3B30;
    --warning: #FFB000;
    --blue: #2F6BFF;
    --royal-blue: #2F6BFF;
  }
  @page { size: A4 landscape; margin: 8mm; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Consolas", monospace, sans-serif;
    font-size: 15px;
    color: var(--text-primary);
    background-color: var(--page-bg);
    margin: 0;
    padding: 24px;
  }
  .container { max-width: 1550px; margin: 0 auto; }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 2px solid #3A4654;
    padding-bottom: 12px;
    margin-bottom: 16px;
  }
  .header h1 { font-size: 26px; margin: 0 0 6px 0; color: var(--text-primary); font-weight: 800; letter-spacing: -0.5px; text-transform: uppercase; }
  .header-sub { font-size: 13px; color: var(--subtitle); }
  .header-meta { text-align: right; font-size: 13px; color: var(--subtitle); }
  .status-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 3px;
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
  }
  .status-SUCCESS { background: rgba(0, 230, 118, 0.15); color: var(--positive); border: 1px solid var(--positive); }
  .status-SUCCESS_WITH_WARNINGS { background: rgba(255, 176, 0, 0.15); color: var(--warning); border: 1px solid var(--warning); }
  .status-PARTIAL { background: rgba(255, 112, 67, 0.15); color: #FF7043; border: 1px solid #FF7043; }
  .status-FAILED { background: rgba(255, 59, 48, 0.15); color: var(--negative); border: 1px solid var(--negative); }

  .kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }
  .kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 14px 16px;
  }
  .kpi-title { font-size: 11.5px; text-transform: uppercase; color: var(--subtitle); font-weight: 700; margin-bottom: 6px; letter-spacing: 0.5px; }
  .kpi-val { font-size: 20px; font-weight: 800; color: var(--text-primary); }

  .card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 18px;
    page-break-inside: avoid;
  }
  .card-title { font-size: 14.5px; font-weight: 800; color: var(--text-primary); margin-bottom: 10px; border-bottom: 1px solid #33404D; padding-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  .card-subtitle { font-size: 12px; color: var(--subtitle); margin-top: -6px; margin-bottom: 12px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .interp-box { background: #13171D; border-left: 3px solid var(--subtitle); padding: 9px 12px; margin-top: 8px; font-size: 12.5px; line-height: 1.45; color: #D9E1EA; border-radius: 0 4px 4px 0; }

  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th, td { padding: 7px 9px; text-align: right; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  th { border-bottom: 1px solid var(--border-color); color: var(--subtitle); font-weight: 700; font-size: 12px; background: var(--card-alt); text-transform: uppercase; }
  td { border-bottom: 1px solid #1c1c1c; font-variant-numeric: tabular-nums; }
  tbody tr:nth-child(even) { background-color: rgba(255, 255, 255, 0.02); }
  .val-up { color: var(--negative); font-weight: bold; }
  .val-down { color: var(--blue); font-weight: bold; }
  .val-neutral { color: var(--text-secondary); }
  .chart-full { width: 100%; border-radius: 4px; display: block; margin-top: 8px; }
  ul.bullets { margin: 0; padding-left: 18px; font-size: 14px; }
  ul.bullets li { margin-bottom: 8px; line-height: 1.45; color: var(--text-primary); }
  .badge-proxy { font-size: 10px; padding: 2px 5px; border-radius: 3px; background: #1f1f1f; border: 1px solid var(--border-color); font-weight: bold; color: var(--warning); }
  .badge-ok { font-size: 10px; padding: 2px 5px; border-radius: 3px; background: rgba(0, 230, 118, 0.12); color: var(--positive); font-weight: bold; border: 1px solid rgba(0, 230, 118, 0.3); }
  .badge-stale { font-size: 10px; padding: 2px 5px; border-radius: 3px; background: rgba(255, 176, 0, 0.12); color: var(--warning); font-weight: bold; border: 1px solid rgba(255, 176, 0, 0.3); }
  .badge-missing { font-size: 10px; padding: 2px 5px; border-radius: 3px; background: rgba(255, 59, 48, 0.12); color: var(--negative); font-weight: bold; border: 1px solid rgba(255, 59, 48, 0.3); }
  .badge-source_error { font-size: 10px; padding: 2px 5px; border-radius: 3px; background: rgba(255, 59, 48, 0.2); color: var(--negative); font-weight: bold; }
  .badge-insufficient_history { font-size: 10px; padding: 2px 5px; border-radius: 3px; background: #262626; color: var(--text-secondary); font-weight: bold; }
  .empty-banner { background: #141414; border: 1px dashed var(--border-color); padding: 16px; border-radius: 4px; color: var(--warning); font-size: 13px; text-align: center; }
  .badge-market-implied { background: rgba(0, 230, 118, 0.15); color: var(--positive); border: 1px solid var(--positive); padding: 3px 8px; border-radius: 3px; font-weight: bold; font-size: 11px; }
  .badge-surface-unavailable { background: rgba(255, 59, 48, 0.15); color: var(--negative); border: 1px solid var(--negative); padding: 3px 8px; border-radius: 3px; font-weight: bold; font-size: 11px; }
  .methodology-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12.5px; }
  .methodology-table th { background: #141820; color: var(--subtitle); padding: 6px 10px; text-align: left; border-bottom: 1px solid #293241; font-weight: 700; }
  .methodology-table td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #1C222B; color: #D9E1EA; }
  .methodology-table tr:nth-child(even) { background: rgba(255, 255, 255, 0.015); }
  .methodology-desc { font-size: 12.5px; line-height: 1.5; color: #D9E1EA; margin-top: 10px; }
  .demo-warning { background: rgba(255, 176, 0, 0.12); border: 1px solid var(--warning); padding: 10px 14px; border-radius: 4px; font-weight: bold; color: var(--warning); margin-bottom: 16px; font-size: 13px; }

  @media print {
    body { background-color: #050505; color: #F2F2F2; padding: 0; font-size: 9.5pt; }
    .container { max-width: 100%; }
    .card { border: 1px solid #333; page-break-inside: avoid; }
    .page-break { page-break-before: always; }
    table { font-size: 8.5pt; }
    th, td { padding: 4px 5px; }
  }
</style>
</head>
<body>
<div class="container">

{% if demo_mode %}
<div class="demo-warning">
  ⚠️ <strong>DEMONSTRATION MODE:</strong> Synthetic or cached evaluation data enabled. Not for live investment decisions.
</div>
{% endif %}

<div class="header">
  <div>
    <h1>{{ headline if headline else "Morning Macro-Financial Risk Monitor" }}</h1>
    <div class="header-sub">Pension Investment Risk Support | Base Currency: <strong>EUR</strong> | Settlement: TARGET & Federal Reserve</div>
  </div>
  <div class="header-meta">
    <div><strong>As of:</strong> {{ as_of_date }} | <strong>Report Status:</strong> <span class="status-badge status-{{ run_status }}">{{ run_status }}</span></div>
    <div style="color: var(--text-secondary); margin-top: 4px;">Executed: {{ execution_time }} ({{ timezone }})</div>
  </div>
</div>

<div class="kpi-strip">
  <div class="kpi-card"><div class="kpi-title">Cross-Asset Regime</div><div class="kpi-val">{{ regime_label }}</div></div>
  <div class="kpi-card"><div class="kpi-title">Yield Curve Dynamics</div><div class="kpi-val">{{ curve_label }}</div></div>
  <div class="kpi-card"><div class="kpi-title">Max Rates Move (1D)</div><div class="kpi-val">{{ max_rates_move }}</div></div>
  <div class="kpi-card"><div class="kpi-title">Data Feed Integrity</div><div class="kpi-val">{{ stale_count }} Stale | {{ missing_count }} Missing</div></div>
</div>

<!-- SECTION: TOP-OF-PAGE MARKET REGIME INTERPRETATION -->
<div class="card">
  <div class="card-title">Market Regime Interpretation</div>
  <div class="card-subtitle">Deterministic multi-asset risk synthesis and yield curve state classification</div>
  
  <div style="margin-bottom: 12px;">
    <div style="font-size: 12px; font-weight: bold; color: var(--subtitle); text-transform: uppercase; margin-bottom: 6px;">Today's Market Summary</div>
    <ul class="bullets">
      {% for s in today_market_summary %}
      <li>{{ s }}</li>
      {% endfor %}
    </ul>
  </div>

  <div class="grid-2" style="margin-top: 14px;">
    <div style="background: #111419; border: 1px solid #202733; border-radius: 4px; padding: 12px;">
      <div style="font-size: 11.5px; font-weight: bold; color: var(--subtitle); text-transform: uppercase; margin-bottom: 4px;">Yield Curve Dynamics: <span style="color: var(--text-primary);">{{ yield_curve_regime.label if yield_curve_regime else curve_label }}</span></div>
      <div style="font-size: 12.5px; color: #D9E1EA; margin-bottom: 6px;">{{ yield_curve_regime.plain_language_definition if yield_curve_regime else '' }}</div>
      <div style="font-size: 11.5px; color: var(--text-secondary);">
        Duration: <strong>{{ yield_curve_regime.duration_sessions if yield_curve_regime else 1 }}</strong> session(s) | 1Y Frequency: <strong>{{ yield_curve_regime.frequency_1y if (yield_curve_regime and yield_curve_regime.frequency_1y is not none) else '-' }}%</strong> | 5D Horizon: <strong>{{ yield_curve_regime.horizon_labels.get('5D', '-') if yield_curve_regime else '-' }}</strong>
      </div>
    </div>
    <div style="background: #111419; border: 1px solid #202733; border-radius: 4px; padding: 12px;">
      <div style="font-size: 11.5px; font-weight: bold; color: var(--subtitle); text-transform: uppercase; margin-bottom: 4px;">Cross-Asset Regime: <span style="color: var(--text-primary);">{{ cross_asset_regime_detail.label if cross_asset_regime_detail else regime_label }}</span></div>
      <div style="font-size: 12.5px; color: #D9E1EA; margin-bottom: 6px;">{{ cross_asset_regime_detail.plain_language_definition if cross_asset_regime_detail else '' }}</div>
      <div style="font-size: 11.5px; color: var(--text-secondary);">
        Composite Score: <strong>{{ "%+.2f"|format(cross_asset_regime_detail.score) if (cross_asset_regime_detail and cross_asset_regime_detail.score is not none) else '-' }}</strong> | Confidence: <strong>{{ cross_asset_regime_detail.confidence if cross_asset_regime_detail else 'MEDIUM' }}</strong>
        {% if cross_asset_regime_detail and cross_asset_regime_detail.component_contributions %}
        | Components: {% for k, v in cross_asset_regime_detail.component_contributions.items() %}<code>{{ k }}: {{ "%+.1f"|format(v) }}</code> {% endfor %}
        {% endif %}
      </div>
    </div>
  </div>
</div>

<!-- SECTION 1: SOVEREIGN BENCHMARK TERM STRUCTURES & RATES MONITOR -->
<div class="card">
  <div class="card-title">Sovereign Benchmark Term Structures (German Bunds, US Treasuries, Finland)</div>
  <div class="card-subtitle">Numeric tenor ordering (3M to 30Y) with ECB AAA and FRED Constant Maturity points</div>
  {% if sovereign_curve_chart_uri %}
  <img class="chart-full" src="{{ sovereign_curve_chart_uri }}" alt="Sovereign Yield Curves">
  <div class="interp-box">{{ sovereign_curve_interpretation }}</div>
  {% else %}
  <div class="empty-banner">⚠️ Data unavailable: Insufficient valid tenor points to construct sovereign curves.</div>
  {% endif %}
</div>

<div class="card">
  <div class="card-title">Sovereign Rates Monitor (Multi-Country Tenor Grid)</div>
  <table>
    <thead>
      <tr>
        <th>Country</th><th>Ccy</th><th>2Y</th><th>5Y</th><th>10Y</th><th>30Y</th><th>50Y</th>
        <th>2s10s</th><th>5s30s</th><th>10Y 1D</th><th>10Y 5D</th><th>10Y 1M</th><th>10Y 3Y Z</th>
        <th>As-Of</th><th>Age</th><th>Provider</th><th>Status</th>
      </tr>
    </thead>
    <tbody>
      {% for r in sovereign_rates_table %}
      <tr>
        <td><strong>{{ r.country_name }}</strong> ({{ r.country_code }})</td>
        <td>{{ r.currency }}</td>
        <td>{{ r.yield_2y }}</td>
        <td>{{ r.yield_5y }}</td>
        <td><strong>{{ r.yield_10y }}</strong></td>
        <td>{{ r.yield_30y }}</td>
        <td>{{ r.yield_50y }}</td>
        <td>{{ r.spread_2s10s }}</td>
        <td>{{ r.spread_5s30s }}</td>
        <td class="{{ 'val-up' if r.chg_10y_1d_val and r.chg_10y_1d_val > 0 else ('val-down' if r.chg_10y_1d_val and r.chg_10y_1d_val < 0 else 'val-neutral') }}">{{ r.chg_10y_1d }}</td>
        <td>{{ r.chg_10y_5d }}</td>
        <td>{{ r.chg_10y_1m }}</td>
        <td><strong>{{ r.zscore_10y_3y }}</strong></td>
        <td style="color: var(--text-secondary);">{{ r.as_of }}</td>
        <td>{{ r.data_age_days }}d</td>
        <td>{{ r.provider }}</td>
        <td><span class="badge-{{ r.status.lower() }}">{{ r.status }}</span></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

<!-- RELATIVE-VALUE SOVEREIGN SPREADS -->
{% if relative_value_spreads %}
<div class="card">
  <div class="card-title">Sovereign Relative Value & Curve Slope Spreads</div>
  <table>
    <thead>
      <tr><th>Relationship / Spread</th><th>Spread (bp)</th><th>1D</th><th>5D</th><th>1M</th><th>Long Leg Date</th><th>Bench Date</th><th>Provider</th><th>Analytical Interpretation</th></tr>
    </thead>
    <tbody>
      {% for s in relative_value_spreads %}
      <tr>
        <td style="text-align: left;"><strong>{{ s.label }}</strong></td>
        <td><strong class="{{ 'val-up' if s.spread_val and s.spread_val > 0 else 'val-down' }}">{{ s.spread_bp }}</strong></td>
        <td>{{ s.chg_1d }}</td>
        <td>{{ s.chg_5d }}</td>
        <td>{{ s.chg_1m }}</td>
        <td>{{ s.long_date }}</td>
        <td>{{ s.benchmark_date }}</td>
        <td>{{ s.provider }}</td>
        <td style="text-align: left; color: var(--text-secondary);">{{ s.commentary }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}

<!-- SECTION 2: RATES & CREDIT SPREAD HISTORIES -->
<div class="page-break"></div>
<div class="card">
  <div class="card-title">Benchmark Yield Rolling 3Y History</div>
  {% if rates_history_chart_uri %}
  <img class="chart-full" src="{{ rates_history_chart_uri }}" alt="Rates History">
  <div class="interp-box">{{ rates_history_interpretation }}</div>
  {% else %}
  <div class="empty-banner">⚠️ Data unavailable: Historical benchmark rates observations not loaded.</div>
  {% endif %}
</div>

<div class="grid-2">
  <div class="card">
    <div class="card-title">Credit Spreads History (US HY & IG OAS)</div>
    {% if credit_history_chart_uri %}
    <img class="chart-full" src="{{ credit_history_chart_uri }}" alt="Credit OAS History">
    <div class="interp-box">{{ credit_history_interpretation }}</div>
    {% else %}
    <div class="empty-banner">⚠️ Data unavailable: FRED credit spread history failed or unconfigured API key.</div>
    {% endif %}
  </div>
  <div class="card">
    <div class="card-title">Cross-Asset Disruption (3Y Z-Score)</div>
    {% if zscore_chart_uri %}
    <img class="chart-full" src="{{ zscore_chart_uri }}" alt="Z-Score Chart">
    <div class="interp-box">{{ zscore_interpretation }}</div>
    {% else %}
    <div class="empty-banner">⚠️ Insufficient history: Minimum 252 daily moves required for z-score calculation.</div>
    {% endif %}
  </div>
</div>

<!-- SECTION 3: MARKET OVERVIEW TABLE & VOLATILITY -->
<div class="card">
  <div class="card-title">Multi-Asset Risk Indicators (Equities, Credit, Volatility, FX)</div>
  <table>
    <thead>
      <tr><th>Asset Class</th><th>Instrument</th><th>Level</th><th>Unit</th><th>1D</th><th>5D</th><th>1M</th><th>YTD</th><th>1Y %tile</th><th>3Y Z-Score</th><th>As-Of</th><th>Status</th></tr>
    </thead>
    <tbody>
      {% for m in market_table %}
      <tr>
        <td><strong>{{ m.country or m.tenor or '-' }}</strong></td>
        <td>{{ m.name }}{% if m.is_proxy %} <span class="badge-proxy">PROXY</span>{% endif %}</td>
        <td><strong>{{ m.level }}</strong></td>
        <td style="color: var(--text-secondary);">{{ m.unit }}</td>
        <td class="{{ 'val-up' if m.is_adverse else 'val-down' }}">{{ m.chg }}</td>
        <td>{{ m.chg_5d }}</td>
        <td>{{ m.chg_1m }}</td>
        <td>{{ m.chg_ytd }}</td>
        <td>{{ m.pctile }}</td>
        <td><strong>{{ m.zscore }}</strong></td>
        <td style="color: var(--text-secondary);">{{ m.as_of }}</td>
        <td><span class="badge-{{ m.status.lower() }}">{{ m.status }}</span></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

<!-- SECTION 3A: STANDALONE EUR/USD AND VIX TIME-SERIES CHARTS -->
<div class="grid-2">
  <div class="card">
    <div class="card-title">EUR/USD Exchange Rate History</div>
    <div class="card-subtitle">USD per EUR | Higher values denote euro appreciation</div>
    {% if eurusd_chart_uri %}
    <img class="chart-full" src="{{ eurusd_chart_uri }}" alt="EUR/USD Exchange Rate History">
    <div class="interp-box">{{ eurusd_interpretation }}</div>
    {% else %}
    <div class="empty-banner">⚠️ Data unavailable: EUR/USD time series failed from ECB Reference Rates and Yahoo fallback.</div>
    {% endif %}
  </div>
  <div class="card">
    <div class="card-title">CBOE Volatility Index (VIX) History</div>
    <div class="card-subtitle">Option-implied expectation of near-term S&P 500 volatility</div>
    {% if vix_chart_uri %}
    <img class="chart-full" src="{{ vix_chart_uri }}" alt="CBOE Volatility Index (VIX) History">
    <div class="interp-box">{{ vix_interpretation }}</div>
    {% else %}
    <div class="empty-banner">⚠️ Data unavailable: CBOE VIX time series failed from FRED and Yahoo fallback.</div>
    {% endif %}
  </div>
</div>

<!-- SECTION 3B: BTC/USD MARKET-IMPLIED VOLATILITY SURFACE -->
<div class="card">
  <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; margin-bottom: 12px;">
    <div class="card-title" style="margin-bottom: 0; border-bottom: none; padding-bottom: 0;">
      BTC/USD Market-Implied Volatility Surface
    </div>
    <div>
      {% if btc_volatility_surface and btc_volatility_surface.status.value in ['AVAILABLE', 'AVAILABLE_WITH_WARNINGS'] %}
      <span class="badge-market-implied">MARKET IMPLIED</span>
      {% else %}
      <span class="badge-surface-unavailable">{{ btc_volatility_surface.status.value if btc_volatility_surface else 'UNAVAILABLE' }}</span>
      {% endif %}
    </div>
  </div>

  <div class="grid-2">
    <div>
      {% if vol_surface_chart_uri %}
      <img class="chart-full" src="{{ vol_surface_chart_uri }}" alt="Volatility Surface Heatmap">
      {% else %}
      <div class="empty-banner">⚠️ BTC Surface Heatmap Unavailable: Deribit options data feed offline or calibration criteria unmet.</div>
      {% endif %}
    </div>
    <div>
      {% if vol_surface_3d_chart_uri %}
      <img class="chart-full" src="{{ vol_surface_3d_chart_uri }}" alt="3D Volatility Surface">
      {% else %}
      <div class="empty-banner">⚠️ 3D BTC Surface View Unavailable.</div>
      {% endif %}
    </div>
  </div>

  <div class="interp-box" style="margin-top: 12px;">{{ btc_surface_interpretation }}</div>

  <!-- BTC SURFACE MODELLING EXPLANATION & METHODOLOGY TABLE -->
  <div style="margin-top: 14px;">
    <div style="font-size: 12.5px; font-weight: bold; color: var(--subtitle); text-transform: uppercase; margin-bottom: 4px;">BTC/USD Volatility Surface Construction Methodology</div>
    <div class="methodology-desc">
      The BTC/USD volatility surface is calibrated directly from active Deribit option-market quotes spanning multiple strikes and expiries ingested via Deribit production API v2. After applying liquidity and spread filters, out-of-the-money (OTM) options are isolated to eliminate put-call disparity. Each expiry slice is fitted in total-variance space $w(k) = \sigma^2 \tau$ using Gatheral's Quasi-Explicit Raw SVI parameterization. Monotone PCHIP cubic splines interpolate across maturity to prevent calendar arbitrage crossings. Visible scatter markers represent genuine exchange option-implied quotes, whereas the continuous mesh consists of calibrated model evaluations. The surface models market-implied uncertainty across strike and maturity and carries no directional price forecast.
    </div>
    <table class="methodology-table">
      <thead>
        <tr><th>Parameter / Field</th><th>Value / Specification</th><th>Diagnostic Status / Description</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>Exchange Data Provider</strong></td><td>DERIBIT API v2 (Unauthenticated Public Endpoint)</td><td>Real-time production book summary and active instrument discovery</td></tr>
        <tr><td><strong>Calibration Architecture</strong></td><td>Raw SVI + Monotone PCHIP Spline</td><td>Gatheral Quasi-Explicit SVI per expiry; PCHIP across maturities</td></tr>
        <tr><td><strong>Ingested vs. Retained Quotes</strong></td><td>{{ btc_volatility_surface.raw_instrument_count if btc_volatility_surface else '-' }} raw / {{ btc_volatility_surface.retained_quote_count if btc_volatility_surface else '-' }} retained</td><td>Filtered by moneyness [0.60, 1.40], maturity [7d, 365d], valid mark IV</td></tr>
        <tr><td><strong>Retained Expiries</strong></td><td>{{ btc_volatility_surface.retained_expiries | length if btc_volatility_surface else '-' }} expiries</td><td>{{ btc_volatility_surface.retained_expiries | join(', ') if btc_volatility_surface else '-' }}</td></tr>
        <tr><td><strong>Weighted Calibration RMSE</strong></td><td><strong>{{ "%.2f%%"|format(btc_volatility_surface.weighted_rmse) if (btc_volatility_surface and btc_volatility_surface.weighted_rmse is not none) else '-' }}</strong></td><td>Liquidity-weighted root-mean-square calibration error across slices</td></tr>
        <tr><td><strong>Dense Model Grid</strong></td><td>{{ btc_volatility_surface.dense_model_point_count if btc_volatility_surface else '-' }} points (60T × 81k)</td><td>Dense mesh coverage: {{ "%.1f%%"|format(btc_volatility_surface.dense_grid_coverage_pct) if btc_volatility_surface else '-' }}</td></tr>
        <tr><td><strong>Arbitrage Compliance</strong></td><td>Calendar: {{ btc_volatility_surface.arbitrage_diagnostics.get('calendar_status', '-') if btc_volatility_surface else '-' }} | Butterfly: {{ btc_volatility_surface.arbitrage_diagnostics.get('butterfly_status', '-') if btc_volatility_surface else '-' }}</td><td>{{ btc_volatility_surface.arbitrage_diagnostics.get('calendar_arbitrage_violations', 0) if btc_volatility_surface else 0 }} calendar / {{ btc_volatility_surface.arbitrage_diagnostics.get('butterfly_arbitrage_violations', 0) if btc_volatility_surface else 0 }} butterfly violations</td></tr>
        <tr><td><strong>Snapshot Timestamp</strong></td><td>{{ btc_volatility_surface.valuation_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') if btc_volatility_surface else '-' }}</td><td>Helsinki Local: {{ btc_volatility_surface.valuation_timestamp_helsinki if btc_volatility_surface else '-' }}</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- SECTION 4: PENSION TRANSMISSION & OFFICIAL RELEASES -->
<div class="grid-2">
  <div class="card">
    <div class="card-title">Pension Portfolio Transmission Narrative</div>
    <ul class="bullets">
      {% for b in narrative_bullets %}
      <li>{{ b }}</li>
      {% endfor %}
    </ul>
  </div>
  <div class="card">
    <div class="card-title">Key Central Bank & Macro Releases</div>
    <ul class="bullets">
      {% for n in news_items %}
      <li><strong>[{{ n.publisher }}]</strong> <a href="{{ n.url }}" target="_blank" style="color: var(--royal-blue); text-decoration: none;">{{ n.title }}</a></li>
      {% endfor %}
    </ul>
  </div>
</div>

<!-- SECTION 6: DATA QUALITY & GOVERNANCE PANEL -->
<div class="card">
  <div class="card-title">Data Quality & Feed Governance</div>
  <table>
    <thead>
      <tr><th>Series ID</th><th>Instrument</th><th>Provider</th><th>Retrieval</th><th>Obs Count</th><th>Latest Obs</th><th>Effective</th><th>Age</th><th>Status</th><th>Diagnostics / Errors</th></tr>
    </thead>
    <tbody>
      {% for item in quality_summary['items'] %}
      <tr>
        <td><code>{{ item.series_id }}</code></td>
        <td>{{ item.display_name }}{% if item.is_demo %} <span class="badge-proxy">DEMO</span>{% endif %}</td>
        <td>{{ item.provider | upper }}</td>
        <td>{{ item.retrieval_status }}</td>
        <td>{{ item.obs_count }}</td>
        <td>{{ item.latest_date }}</td>
        <td>{{ item.effective_date or '-' }}</td>
        <td>{{ item.stale_days }} d</td>
        <td><span class="badge-{{ item.status.lower() }}">{{ item.status }}</span></td>
        <td style="font-size: 12px; color: var(--text-secondary); text-align: left;">
          {% if item.error_detail %}<span style="color: var(--negative);">{{ item.error_detail }}</span>{% elif item.errors %}<span style="color: var(--negative);">{{ item.errors | join('; ') }}</span>{% elif item.warnings %}<span style="color: var(--warning);">{{ item.warnings | join('; ') }}</span>{% else %}<span style="color: var(--positive);">Clean observation feed</span>{% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

<!-- SECTION 7: METHODOLOGY & AUTHORITATIVE REFERENCE CATALOGUE -->
<div class="card">
  <div class="card-title">Methodology & Authoritative Reference Sources</div>
  <div style="font-size: 12.5px; color: #D9E1EA; line-height: 1.5;">
    <ul class="bullets">
      <li><strong>Yield Curve Regimes & 2s10s Dynamics:</strong> Defined per Federal Reserve and ECB term-structure literature. Bear/Bull flattening and steepening isolate short-end policy shifts from long-end term premium expansion.</li>
      <li><strong>Corporate Credit OAS:</strong> Based on ICE Data Indices / Federal Reserve Bank of St. Louis FRED methodology. Option-adjusted spreads remove embedded call/put optionality from cash bond yields.</li>
      <li><strong>CBOE Volatility Index (VIX):</strong> Formulated per Cboe Global Markets white papers as the annualized 30-day square root of expected variance across out-of-the-money S&P 500 options. Context zones (subdued &lt;15, moderate 15–20, elevated 20–30, stress &gt;30) represent heuristic benchmarks.</li>
      <li><strong>Foreign Exchange Convention:</strong> Quoted strictly as USD per EUR in accordance with ECB reference rates; rising values reflect euro appreciation.</li>
      <li><strong>BTC Volatility Surface Calibration:</strong> Quasi-Explicit Raw SVI parameterization per Gatheral (2004) with monotone cubic spline interpolation in total variance space. Ingested via Deribit production API v2.</li>
    </ul>
  </div>
</div>

</div>
</body>
</html>
"""


class ReportRenderer:
    @staticmethod
    def render_html(context: ReportContext) -> str:
        template = jinja2.Template(HTML_TEMPLATE)
        return template.render(**context.model_dump())

    @staticmethod
    def render_pdf(html_text: str, target_path: Path) -> None:
        try:
            import weasyprint
            weasyprint.HTML(string=html_text).write_pdf(str(target_path))
            if target_path.exists() and target_path.stat().st_size > 0:
                with open(target_path, "rb") as f:
                    if f.read(5).startswith(b"%PDF"):
                        return
            if target_path.exists():
                target_path.unlink()
        except Exception as exc:
            if target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass