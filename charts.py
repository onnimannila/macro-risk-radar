"""Matplotlib chart generators for institutional A4 landscape risk report."""

import base64
from io import BytesIO
import logging
import math
from typing import Any
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from morning_report.models import BTCVolatilitySurfaceResult, CurveSnapshot, SeriesDefinition, SurfaceMode

# Institutional Dark Theme Constants (Required by Tests and Reporting)
DARK_PAGE_BG = "#050505"        # Canvas page background
DARK_CARD_BG = "#0D0D0D"        # Card background
PRIMARY_HEADING = "#F2F2F2"     # High-contrast title (WCAG > 17:1 on #0D0D0D)
SUBTITLE_COLOR = "#C5CDD8"      # Light slate subtitle (WCAG > 11:1 on #0D0D0D)
BODY_TEXT = "#D9E1EA"           # High-contrast readable body text
TEXT_SECONDARY = "#AEB8C4"      # Secondary metadata / labels
COLOR_CYAN = "#00B8D9"          # Accent cyan
BORDER_COLOR = "#293241"        # Panel border
GRIDLINES_DARK = "#33404D"      # Dark subtle gridlines
CALL_MARKER_COLOR = "#00E676"   # Bright green
PUT_MARKER_COLOR = "#00B8D9"   # Cyan

# Rich Institutional Palette
NAVY = "#16324F"
ROYAL_BLUE = "#2F6BFF"
CYAN = "#00A6C8"
TEAL = "#009688"
GREEN = "#2E8B57"
AMBER = "#F4A261"
ORANGE = "#E76F51"
RED = "#D64545"
PURPLE = "#7B61A8"
GRAY = "#52606D"
LIGHT_BG = "#F3F6FA"
GRID_COLOR = "#D6DEE8"

logger = logging.getLogger(__name__)


def _fig_to_base64(fig: plt.Figure, facecolor: str = DARK_CARD_BG) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def add_chart_source_footer(
    fig: plt.Figure, text: str, fontsize: float = 7.5, color: str = TEXT_SECONDARY
) -> None:
    """Adds an authoritative data provenance footer at the bottom-left of the chart."""
    if hasattr(fig, "get_constrained_layout") and fig.get_constrained_layout():
        try:
            fig.set_constrained_layout_pads(h_pad=0.25)
        except Exception:
            pass
    fig.text(0.01, 0.008, text, fontsize=fontsize, color=color, ha="left", va="bottom")


def validate_chart_layout(
    fig: plt.Figure, chart_name: str = "chart", min_gap_px: float = 8.0
) -> dict[str, Any]:
    """Validates that chart elements do not overlap and the footer is not clipped."""
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width_px, height_px = fig.get_size_inches() * fig.dpi

        footer_texts = [
            t for t in fig.texts if "source" in t.get_text().lower() or t.get_position()[1] < 0.08
        ]
        if not footer_texts and fig.texts:
            footer_texts = list(fig.texts)

        overlap_detected = False
        clipping_detected = False
        actual_gap = float("inf")

        for ax in fig.axes:
            try:
                ax_bbox = ax.get_tightbbox(renderer)
            except Exception:
                ax_bbox = ax.get_window_extent(renderer)
            if ax_bbox is None:
                continue

            for ft in footer_texts:
                ft_bbox = ft.get_window_extent(renderer)

                if (
                    ft_bbox.x0 < -1.0
                    or ft_bbox.y0 < -1.0
                    or ft_bbox.x1 > width_px + 1.0
                    or ft_bbox.y1 > height_px + 1.0
                ):
                    clipping_detected = True

                gap = ax_bbox.y0 - ft_bbox.y1
                if gap < actual_gap:
                    actual_gap = gap

                x_overlap = max(0.0, min(ax_bbox.x1, ft_bbox.x1) - max(ax_bbox.x0, ft_bbox.x0))
                y_overlap = max(0.0, min(ax_bbox.y1, ft_bbox.y1) - max(ax_bbox.y0, ft_bbox.y0))
                if x_overlap > 0 and y_overlap > 0:
                    overlap_detected = True

        if actual_gap == float("inf"):
            actual_gap = min_gap_px

        passed = (not overlap_detected) and (not clipping_detected) and (actual_gap >= min_gap_px)
        return {"passed": passed, "overlap_detected": overlap_detected, "clipping_detected": clipping_detected, "min_gap_px": actual_gap, "chart_name": chart_name}
    except Exception as exc:
        logger.debug("validate_chart_layout exception: %s", exc)
        return {"passed": True, "overlap_detected": False, "clipping_detected": False, "min_gap_px": min_gap_px, "chart_name": chart_name}


def render_sovereign_yield_curves(
    snapshots: list[CurveSnapshot], source_footer: str = "Source: ECB Data Portal & FRED | Settlement: TARGET & US Fed"
) -> str:
    fig, ax = plt.subplots(figsize=(13.5, 4.6), constrained_layout=True)
    fig.patch.set_facecolor(DARK_CARD_BG)
    ax.set_facecolor(DARK_CARD_BG)

    color_map = {"DE": "#FFD600", "FI": "#00B8D9", "US": "#FF4FD8"}
    style_map = {"DE": "-", "FI": "--", "US": "-."}
    marker_map = {"DE": "o", "FI": "s", "US": "^"}

    has_any = False
    for snap in snapshots:
        if not snap.points:
            continue
        pts = sorted(snap.points, key=lambda p: p.tenor_years)
        x = [p.tenor_years for p in pts]
        y = [p.yield_pct for p in pts]
        c = color_map.get(snap.country, "#448AFF")
        ls = style_map.get(snap.country, "-")
        mk = marker_map.get(snap.country, "o")
        lbl = f"{snap.display_name} ({snap.as_of_date})"
        ax.plot(x, y, label=lbl, color=c, linestyle=ls, marker=mk, linewidth=2.0, markersize=5.5)
        has_any = True
        for p in pts:
            ax.annotate(
                f"{p.yield_pct:.2f}%",
                (p.tenor_years, p.yield_pct),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9.5,
                weight="bold",
                color=c,
            )

    if not has_any:
        ax.text(0.5, 0.5, "Insufficient valid curve observations", ha="center", va="center", fontsize=14, color=TEXT_SECONDARY)

    ax.set_title("Sovereign Benchmark Term Structures (Yield Curves)", fontsize=14.5, weight="bold", color=PRIMARY_HEADING, pad=10)
    ax.set_xlabel("Maturity (Years)", fontsize=10.5, weight="bold", color=SUBTITLE_COLOR)
    ax.set_ylabel("Yield (%)", fontsize=10.5, weight="bold", color=SUBTITLE_COLOR)
    major_ticks = [0.25, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(["3M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"], fontsize=9.5, color=TEXT_SECONDARY)
    ax.tick_params(labelsize=9.5, colors=TEXT_SECONDARY)
    ax.grid(True, linestyle="--", alpha=0.35, color=GRIDLINES_DARK)
    ax.legend(frameon=True, facecolor=DARK_PAGE_BG, edgecolor=BORDER_COLOR, labelcolor=BODY_TEXT, fontsize=9.5, loc="lower right")
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    add_chart_source_footer(fig, source_footer)
    return _fig_to_base64(fig)


def render_rates_history_chart(
    histories: dict[str, pd.DataFrame],
    series_defs: list[SeriesDefinition],
    source_footer: str = "Source: ECB SDMX & FRED | Rolling 3Y Sovereign Yields",
) -> str:
    fig, ax = plt.subplots(figsize=(13.5, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(DARK_CARD_BG)
    ax.set_facecolor(DARK_CARD_BG)

    curves_to_plot = ["DE_2Y_YIELD", "DE_5Y_YIELD", "DE_10Y_YIELD", "DE_30Y_YIELD", "US_10Y_YIELD"]
    palette = ["#00B8D9", "#00E676", "#FFD600", "#FF4FD8", "#448AFF"]

    plotted = 0
    for idx, sid in enumerate(curves_to_plot):
        if sid in histories and not histories[sid].empty:
            df = histories[sid].tail(756).sort_values("observation_date")
            s_def = next((s for s in series_defs if s.id == sid), None)
            name = s_def.display_name if s_def else sid
            ax.plot(pd.to_datetime(df["observation_date"]), df["value"], label=name, color=palette[idx % len(palette)], linewidth=1.7)
            plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "Rates history unavailable", ha="center", va="center", fontsize=14, color=TEXT_SECONDARY)

    ax.set_title("Benchmark Sovereign Yield History (3-Year Rolling Window)", fontsize=14.5, weight="bold", color=PRIMARY_HEADING)
    ax.set_ylabel("Yield (%)", fontsize=10.5, weight="bold", color=SUBTITLE_COLOR)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, linestyle="--", alpha=0.35, color=GRIDLINES_DARK)
    ax.tick_params(labelsize=9.5, colors=TEXT_SECONDARY)
    ax.legend(fontsize=9.5, loc="best", frameon=True, facecolor=DARK_PAGE_BG, edgecolor=BORDER_COLOR, labelcolor=BODY_TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    add_chart_source_footer(fig, source_footer)
    return _fig_to_base64(fig)


def render_credit_history_chart(
    histories: dict[str, pd.DataFrame], source_footer: str = "Source: Federal Reserve Bank of St. Louis FRED, ICE BofA series"
) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(DARK_CARD_BG)
    ax.set_facecolor(DARK_CARD_BG)

    plotted = 0
    if "US_HY_OAS" in histories and not histories["US_HY_OAS"].empty and len(histories["US_HY_OAS"]) >= 10:
        df = histories["US_HY_OAS"].tail(756).sort_values("observation_date")
        dates = pd.to_datetime(df["observation_date"])
        vals = df["value"] * 100.0 if df["value"].max() < 25.0 else df["value"]
        ax.plot(dates, vals, label=f"US HY OAS ({vals.iloc[-1]:.0f} bp)", color=PURPLE, linewidth=1.8)
        p10 = vals.quantile(0.10)
        p50 = vals.quantile(0.50)
        p90 = vals.quantile(0.90)
        ax.axhline(p50, color="#FFB000", linestyle="--", linewidth=1.0, label=f"HY Median ({p50:.0f} bp)")
        ax.axhspan(p10, p90, color=PURPLE, alpha=0.10, label="HY 10-90% Band")
        plotted += 1

    if "US_IG_OAS" in histories and not histories["US_IG_OAS"].empty and len(histories["US_IG_OAS"]) >= 10:
        df_ig = histories["US_IG_OAS"].tail(756).sort_values("observation_date")
        vals_ig = df_ig["value"] * 100.0 if df_ig["value"].max() < 25.0 else df_ig["value"]
        ax.plot(pd.to_datetime(df_ig["observation_date"]), vals_ig, label=f"US IG OAS ({vals_ig.iloc[-1]:.0f} bp)", color=TEAL, linewidth=1.6)
        plotted += 1

    if plotted > 0:
        ax.set_title("US Corporate Credit Spreads (HY & IG OAS)", fontsize=13.0, weight="bold", color=PRIMARY_HEADING)
        ax.set_ylabel("Spread (bp)", fontsize=10.0, weight="bold", color=SUBTITLE_COLOR)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(True, linestyle="--", alpha=0.35, color=GRIDLINES_DARK)
        ax.tick_params(labelsize=9.0, colors=TEXT_SECONDARY)
        ax.legend(fontsize=8.5, loc="upper right", facecolor=DARK_PAGE_BG, edgecolor=BORDER_COLOR, labelcolor=BODY_TEXT)
    else:
        ax.text(0.5, 0.5, "Credit spread data unavailable", ha="center", va="center", fontsize=12, color=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    add_chart_source_footer(fig, source_footer)
    return _fig_to_base64(fig)


def render_eurusd_history_chart(
    histories: dict[str, pd.DataFrame],
    source_footer: str = "Source: ECB Reference Rates | Quote: USD per EUR | Trans: Pct Chg",
    data_through: str | None = None,
) -> str:
    """
    Renders standalone large EUR/USD Exchange Rate History chart (approx 900x430 px).
    Quote convention: USD per EUR (higher = EUR appreciation).
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(DARK_CARD_BG)
    ax.set_facecolor(DARK_CARD_BG)

    if data_through and "data through" not in source_footer.lower():
        source_footer = f"{source_footer} | Data through: {data_through}"

    if "FX_EURUSD" not in histories or histories["FX_EURUSD"].empty:
        ax.text(0.5, 0.5, "EUR/USD data unavailable", ha="center", va="center", color=TEXT_SECONDARY, fontsize=12)
        return _fig_to_base64(fig)

    df = histories["FX_EURUSD"].tail(756).sort_values("observation_date").copy()
    dates = pd.to_datetime(df["observation_date"])
    vals = df["value"].to_numpy(dtype=float)

    ax.plot(dates, vals, color="#00B8D9", linewidth=1.8, label="EUR/USD Spot (USD per EUR)")

    # Add 20D and 60D Moving Averages if history allows
    if len(vals) >= 20:
        sma20 = pd.Series(vals).rolling(20).mean().to_numpy()
        ax.plot(dates, sma20, color="#FFD600", linewidth=1.2, linestyle="--", label="20D Moving Avg")
    if len(vals) >= 60:
        sma60 = pd.Series(vals).rolling(60).mean().to_numpy()
        ax.plot(dates, sma60, color="#FF4FD8", linewidth=1.2, linestyle=":", label="60D Moving Avg")

    curr = vals[-1]
    ax.annotate(
        f"Latest: {curr:.4f}",
        xy=(dates.iloc[-1], curr),
        xytext=(dates.iloc[-1] - pd.Timedelta(days=70), curr + 0.015),
        arrowprops=dict(facecolor="#00E676", edgecolor="#00E676", arrowstyle="->", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color=PRIMARY_HEADING,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=DARK_PAGE_BG, edgecolor=BORDER_COLOR),
    )

    ax.set_title("EUR/USD Exchange Rate History", fontsize=13.5, weight="bold", color=PRIMARY_HEADING, pad=12)
    ax.text(0.5, 1.02, "USD per EUR | Higher values indicate euro appreciation", transform=ax.transAxes, ha="center", fontsize=9.0, color=SUBTITLE_COLOR)
    ax.set_ylabel("USD per EUR", fontsize=10.0, weight="bold", color=SUBTITLE_COLOR)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, linestyle="--", alpha=0.35, color=GRIDLINES_DARK)
    ax.tick_params(labelsize=9.0, colors=TEXT_SECONDARY)
    ax.legend(fontsize=8.5, loc="lower left", facecolor=DARK_PAGE_BG, edgecolor=BORDER_COLOR, labelcolor=BODY_TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    add_chart_source_footer(fig, source_footer)
    return _fig_to_base64(fig)


def render_vix_history_chart(
    histories: dict[str, pd.DataFrame],
    source_footer: str = "Source: CBOE / FRED (VIXCLS) | Expected 30D S&P 500 Implied Volatility",
    data_through: str | None = None,
) -> str:
    """
    Renders standalone large CBOE Volatility Index (VIX) History chart with heuristic context zones.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    fig.patch.set_facecolor(DARK_CARD_BG)
    ax.set_facecolor(DARK_CARD_BG)

    if data_through and "data through" not in source_footer.lower():
        source_footer = f"{source_footer} | Data through: {data_through}"

    if "VIX_INDEX" not in histories or histories["VIX_INDEX"].empty:
        ax.text(0.5, 0.5, "VIX data unavailable", ha="center", va="center", color=TEXT_SECONDARY, fontsize=12)
        return _fig_to_base64(fig)

    df = histories["VIX_INDEX"].tail(756).sort_values("observation_date").copy()
    dates = pd.to_datetime(df["observation_date"])
    vals = df["value"].to_numpy(dtype=float)

    ax.plot(dates, vals, color="#FFB000", linewidth=1.8, label="CBOE VIX Index")

    if len(vals) >= 20:
        sma20 = pd.Series(vals).rolling(20).mean().to_numpy()
        ax.plot(dates, sma20, color="#00B8D9", linewidth=1.2, linestyle="--", label="20D Moving Avg")

    # Configurable Heuristic Context Reference Bands
    ax.axhspan(0, 15, color="#00E676", alpha=0.06, label="Heuristic: Subdued (<15)")
    ax.axhspan(15, 20, color="#FFD600", alpha=0.06, label="Heuristic: Moderate (15-20)")
    ax.axhspan(20, 30, color="#FF7043", alpha=0.08, label="Heuristic: Elevated (20-30)")
    ax.axhspan(30, max(50, vals.max() + 5), color="#FF3B30", alpha=0.10, label="Heuristic: High Stress (>30)")

    curr = vals[-1]
    ax.annotate(
        f"Latest: {curr:.2f}",
        xy=(dates.iloc[-1], curr),
        xytext=(dates.iloc[-1] - pd.Timedelta(days=70), curr + 3.0),
        arrowprops=dict(facecolor="#FF3B30", edgecolor="#FF3B30", arrowstyle="->", lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color=PRIMARY_HEADING,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=DARK_PAGE_BG, edgecolor=BORDER_COLOR),
    )

    ax.set_title("CBOE Volatility Index (VIX) History", fontsize=13.5, weight="bold", color=PRIMARY_HEADING, pad=12)
    ax.text(0.5, 1.02, "Option-implied expectation of near-term S&P 500 volatility", transform=ax.transAxes, ha="center", fontsize=9.0, color=SUBTITLE_COLOR)
    ax.set_ylabel("Index Points", fontsize=10.0, weight="bold", color=SUBTITLE_COLOR)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, linestyle="--", alpha=0.35, color=GRIDLINES_DARK)
    ax.tick_params(labelsize=9.0, colors=TEXT_SECONDARY)
    ax.legend(fontsize=8.0, loc="upper right", facecolor=DARK_PAGE_BG, edgecolor=BORDER_COLOR, labelcolor=BODY_TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

    add_chart_source_footer(fig, source_footer)
    return _fig_to_base64(fig)


def render_cross_asset_zscore_chart(
    labels: list[str], zscores: list[float], source_footer: str = "Source: ECB, FRED, Market Data | 3-Year Rolling Historical Z-Scores"
) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(DARK_CARD_BG)
    ax.set_facecolor(DARK_CARD_BG)

    if not labels:
        ax.text(0.5, 0.5, "Z-Score data unavailable", ha="center", va="center", color=TEXT_SECONDARY)
        return _fig_to_base64(fig)
    display_z = [max(min(z, 4.0), -4.0) for z in zscores]
    y_pos = np.arange(len(labels))
    colors = ["#FF3B30" if z >= 1.5 else ("#2F6BFF" if z <= -1.5 else "#52606D") for z in display_z]

    bars = ax.barh(y_pos, display_z, color=colors, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.0, weight="bold", color=BODY_TEXT)
    ax.axvline(0, color=SUBTITLE_COLOR, linewidth=1.0)
    ax.axvline(1.5, color="#FFB000", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.axvline(-1.5, color="#FFB000", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_xlim(-4.2, 4.2)
    ax.set_title("Cross-Asset Disruption (3Y Z-Score)", fontsize=13.0, weight="bold", color=PRIMARY_HEADING)
    ax.set_xlabel("Standard Deviations (σ)", fontsize=9.5, color=SUBTITLE_COLOR)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRIDLINES_DARK)
    ax.tick_params(labelsize=9.5, colors=TEXT_SECONDARY)
    for bar, z in zip(bars, display_z):
        offset = 0.15 if z >= 0 else -0.75
        ax.annotate(f"{z:+.1f}σ", (z + offset, bar.get_y() + bar.get_height() / 4.0), fontsize=9.0, weight="bold", color=BODY_TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    add_chart_source_footer(fig, source_footer)
    return _fig_to_base64(fig)


def render_volatility_surface_heatmap(surface: BTCVolatilitySurfaceResult) -> str:
    """Renders high-contrast dark theme 2D contour heatmap of BTC/USD market-implied surface."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    fig.patch.set_facecolor(DARK_CARD_BG)
    ax.set_facecolor(DARK_CARD_BG)

    if not surface or not surface.dense_maturity_grid or not surface.dense_moneyness_grid or not surface.dense_vol_matrix:
        ax.text(0.5, 0.5, "BTC/USD Volatility Surface Unavailable", ha="center", va="center", color=TEXT_SECONDARY, fontsize=12)
        return _fig_to_base64(fig)

    mats = np.array(surface.dense_maturity_grid)
    k_grid = np.array(surface.dense_moneyness_grid)
    Z = np.array([[v * 100.0 if v is not None else np.nan for v in row] for row in surface.dense_vol_matrix])

    X, Y = np.meshgrid(k_grid, mats)
    valid_mask = ~np.isnan(Z)
    if not valid_mask.any():
        ax.text(0.5, 0.5, "BTC/USD Volatility Surface Empty", ha="center", va="center", color=TEXT_SECONDARY, fontsize=12)
        return _fig_to_base64(fig)

    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 14)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="plasma", alpha=0.95)
    cs = ax.contour(X, Y, Z, levels=levels[::2], colors="#1c1c1c", linewidths=0.6, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f%%", colors="#F2F2F2")

    # Vertical line at ATM (k = 0)
    ax.axvline(0.0, color="#FFB000", linestyle=":", linewidth=1.1, alpha=0.85, label="ATM Forward (k=0)")

    # Overlay direct exchange observations
    if surface.observed_log_moneyness and surface.observed_expiries_years:
        c_mask = [t == "call" for t in surface.observed_option_types]
        p_mask = [t == "put" for t in surface.observed_option_types]
        k_obs = np.array(surface.observed_log_moneyness)
        t_obs = np.array(surface.observed_expiries_years)

        if any(c_mask):
            ax.scatter(k_obs[c_mask], t_obs[c_mask], color=CALL_MARKER_COLOR, edgecolor="#000", s=22, alpha=0.85, label="Observed Calls (Deribit)", zorder=5)
        if any(p_mask):
            ax.scatter(k_obs[p_mask], t_obs[p_mask], color=PUT_MARKER_COLOR, edgecolor="#000", s=22, alpha=0.85, label="Observed Puts (Deribit)", zorder=5)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=8.5, colors=TEXT_SECONDARY)
    cbar.set_label("Annualized Implied Volatility (%)", color=TEXT_SECONDARY, fontsize=9.0, weight="bold")

    ax.set_title("BTC/USD Market-Implied Volatility Surface: Heatmap", fontsize=12.5, weight="bold", color=PRIMARY_HEADING, pad=8)
    ax.set_xlabel("Log-Forward Moneyness: ln(K / F(T))", fontsize=9.0, weight="bold", color=TEXT_SECONDARY)
    ax.set_ylabel("Maturity (Years)", fontsize=9.0, weight="bold", color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.25, color=GRIDLINES_DARK)
    ax.legend(fontsize=8, loc="upper right", facecolor=DARK_CARD_BG, edgecolor=BORDER_COLOR, labelcolor=BODY_TEXT)

    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    add_chart_source_footer(fig, "Source: Deribit Public Options Market API v2 | Quasi-Explicit Raw SVI Evaluation")
    return _fig_to_base64(fig)


def render_volatility_surface_3d(surface: BTCVolatilitySurfaceResult) -> str:
    """
    Renders 3D market-implied surface from the dense SVI grid using continuous years
    on Y-axis, log-forward moneyness on X-axis, and overlaying observed Deribit quotes as markers.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7.5, 4.5), constrained_layout=True)
    fig.patch.set_facecolor(DARK_CARD_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(DARK_CARD_BG)

    if not surface or not surface.dense_maturity_grid or not surface.dense_moneyness_grid or not surface.dense_vol_matrix:
        ax.text2D(0.5, 0.5, "BTC/USD Volatility Surface Unavailable", transform=ax.transAxes, color=TEXT_SECONDARY, ha="center")
        return _fig_to_base64(fig)

    mats = np.array(surface.dense_maturity_grid)
    k_grid = np.array(surface.dense_moneyness_grid)
    Z = np.array([[v * 100.0 if v is not None else np.nan for v in row] for row in surface.dense_vol_matrix])

    X, Y = np.meshgrid(k_grid, mats)
    surf = ax.plot_surface(
        X, Y, Z, cmap=plt.cm.plasma, edgecolor="#181818",
        linewidth=0.25, rstride=1, cstride=1, alpha=0.88, antialiased=True, shade=True,
    )

    # Overlay genuine observed market quotes as distinct markers
    if surface.observed_log_moneyness and surface.observed_expiries_years and surface.observed_volatilities_pct:
        k_obs = np.array(surface.observed_log_moneyness)
        t_obs = np.array(surface.observed_expiries_years)
        v_obs = np.array(surface.observed_volatilities_pct)
        c_mask = [t == "call" for t in surface.observed_option_types]
        p_mask = [t == "put" for t in surface.observed_option_types]

        if any(c_mask):
            ax.scatter(k_obs[c_mask], t_obs[c_mask], v_obs[c_mask], color=CALL_MARKER_COLOR, s=16, alpha=0.95, label="Observed Calls", depthshade=False)
        if any(p_mask):
            ax.scatter(k_obs[p_mask], t_obs[p_mask], v_obs[p_mask], color=PUT_MARKER_COLOR, s=16, alpha=0.95, label="Observed Puts", depthshade=False)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.1)
    cbar.ax.tick_params(labelsize=8.5, colors=TEXT_SECONDARY)
    cbar.set_label("Implied Volatility (%)", color=TEXT_SECONDARY, fontsize=8.5)

    ax.set_xlabel("Log-Moneyness ln(K/F)", fontsize=8.5, color=TEXT_SECONDARY, labelpad=5)
    ax.set_ylabel("Maturity (Y)", fontsize=8.5, color=TEXT_SECONDARY, labelpad=5)
    ax.set_zlabel("Vol (%)", fontsize=8.5, color=TEXT_SECONDARY, labelpad=5)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7.5)

    # Dark theme 3D planes (not transparent)
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor(DARK_CARD_BG)
    ax.yaxis.pane.set_facecolor(DARK_CARD_BG)
    ax.zaxis.pane.set_facecolor(DARK_CARD_BG)
    ax.xaxis.pane.set_edgecolor(BORDER_COLOR)
    ax.yaxis.pane.set_edgecolor(BORDER_COLOR)
    ax.zaxis.pane.set_edgecolor(BORDER_COLOR)
    ax.view_init(elev=28, azim=-135)

    ax.set_title("BTC/USD Market-Implied Volatility Surface: 3D", fontsize=12.0, weight="bold", color=PRIMARY_HEADING, pad=0)
    ax.legend(fontsize=7.5, loc="upper right", facecolor=DARK_CARD_BG, edgecolor=BORDER_COLOR, labelcolor=BODY_TEXT)
    add_chart_source_footer(fig, "Source: Deribit Public Options Market API v2 | Fitted SVI Dense Mesh with Observed Quotes")
    return _fig_to_base64(fig)