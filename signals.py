"""Cross-asset regime and multi-horizon yield curve dynamics classification."""

from datetime import date
import numpy as np
import pandas as pd
from morning_report.config import ThresholdConfig
from morning_report.models import CalculatedMetric, RegimeClassification, YieldCurveRegime


class SignalEngine:
    def __init__(self, thresholds: ThresholdConfig) -> None:
        self.thresholds = thresholds

    def classify_yield_curve(
        self,
        delta_2y_bp: float | None,
        delta_10y_bp: float | None,
        histories: dict[str, pd.DataFrame] | None = None,
        target_date: date | None = None,
    ) -> YieldCurveRegime:
        """
        Multi-horizon classification of sovereign curve dynamics with exact definitions and evidence.
        """
        if delta_2y_bp is None or delta_10y_bp is None or np.isnan(delta_2y_bp) or np.isnan(delta_10y_bp):
            return YieldCurveRegime(
                label="Unclassified",
                plain_language_definition="Insufficient short- or long-end sovereign yield data to classify curve dynamics.",
                status="INSUFFICIENT_DATA",
            )

        min_move = 1.0  # bp materiality threshold
        slope_delta = delta_10y_bp - delta_2y_bp
        abs_2y = abs(delta_2y_bp)
        abs_10y = abs(delta_10y_bp)

        if abs_2y < min_move and abs_10y < min_move and abs(slope_delta) < min_move:
            label = "Unchanged / Range-Bound"
            defn = f"Both 2Y ({delta_2y_bp:+.1f} bp) and 10Y ({delta_10y_bp:+.1f} bp) yields moved by less than {min_move:.1f} bp."
            risk = "Yield curve shape invariant over session; minimal duration or convexity rebalancing pressure."
        elif abs(delta_2y_bp - delta_10y_bp) < 1.0 and abs_2y >= min_move:
            label = "Parallel Upward Shift" if delta_10y_bp > 0 else "Parallel Downward Shift"
            defn = f"Uniform shift across maturities: 2Y moved {delta_2y_bp:+.1f} bp, 10Y moved {delta_10y_bp:+.1f} bp (slope change: {slope_delta:+.1f} bp)."
            risk = "Parallel rate shift directly impacts liability discount rates without material curve twist risk."
        elif delta_2y_bp > 0 and delta_10y_bp > 0:
            if slope_delta > 0:
                label = "Bear Steepening"
                defn = f"Both yields rose, with 10Y ({delta_10y_bp:+.1f} bp) outpacing 2Y ({delta_2y_bp:+.1f} bp), steepening the curve by {slope_delta:+.1f} bp."
                risk = "Driven by inflation expectations or term premium expansion; longer-duration assets underperform."
            else:
                label = "Bear Flattening"
                defn = f"Both yields rose, with 2Y ({delta_2y_bp:+.1f} bp) outpacing 10Y ({delta_10y_bp:+.1f} bp), flattening the curve by {abs(slope_delta):.1f} bp."
                risk = "Driven by monetary tightening expectations; policy-sensitive front end leads increases."
        elif delta_2y_bp < 0 and delta_10y_bp < 0:
            if slope_delta > 0:
                label = "Bull Steepening"
                defn = f"Both yields fell, with 2Y ({delta_2y_bp:+.1f} bp) falling more than 10Y ({delta_10y_bp:+.1f} bp), steepening the curve by {slope_delta:+.1f} bp."
                risk = "Typically accompanies central bank rate cuts or flight to front-end liquidity during growth slowdowns."
            else:
                label = "Bull Flattening"
                defn = f"Both yields fell, with 10Y ({delta_10y_bp:+.1f} bp) falling more than 2Y ({delta_2y_bp:+.1f} bp), flattening the curve by {abs(slope_delta):.1f} bp."
                risk = "Long end rallies strongly on declining long-term growth/inflation expectations."
        else:
            if slope_delta > 0:
                label = "Mixed Steepening"
                defn = f"Divergent shifts: 2Y moved {delta_2y_bp:+.1f} bp, 10Y moved {delta_10y_bp:+.1f} bp, steepening 2s10s by {slope_delta:+.1f} bp."
                risk = "Asymmetric repricing along the curve; barbell portfolios experience active divergence."
            else:
                label = "Mixed Flattening"
                defn = f"Divergent shifts: 2Y moved {delta_2y_bp:+.1f} bp, 10Y moved {delta_10y_bp:+.1f} bp, flattening 2s10s by {abs(slope_delta):.1f} bp."
                risk = "Front-end rises while long end falls; classic pre-recession or aggressive policy hike signal."

        evidence = [
            f"2Y yield move: {delta_2y_bp:+.1f} bp",
            f"10Y yield move: {delta_10y_bp:+.1f} bp",
            f"2s10s slope move: {slope_delta:+.1f} bp",
        ]

        # Multi-horizon classifications if histories supplied
        horizon_labels = {"1D": label}
        duration_sessions = 1
        frequency_1y = None
        if histories and "DE_2Y_YIELD" in histories and "DE_10Y_YIELD" in histories:
            df2 = histories["DE_2Y_YIELD"].sort_values("observation_date")
            df10 = histories["DE_10Y_YIELD"].sort_values("observation_date")
            merged = pd.merge(df2, df10, on="observation_date", suffixes=("_2y", "_10y")).dropna()
            if len(merged) >= 6:
                d2_5d = (merged["value_2y"].iloc[-1] - merged["value_2y"].iloc[-6]) * 100.0
                d10_5d = (merged["value_10y"].iloc[-1] - merged["value_10y"].iloc[-6]) * 100.0
                reg_5d = self.classify_yield_curve(d2_5d, d10_5d)
                horizon_labels["5D"] = reg_5d.label
            if len(merged) >= 22:
                d2_1m = (merged["value_2y"].iloc[-1] - merged["value_2y"].iloc[-22]) * 100.0
                d10_1m = (merged["value_10y"].iloc[-1] - merged["value_10y"].iloc[-22]) * 100.0
                reg_1m = self.classify_yield_curve(d2_1m, d10_1m)
                horizon_labels["1M"] = reg_1m.label
            if len(merged) >= 64:
                d2_3m = (merged["value_2y"].iloc[-1] - merged["value_2y"].iloc[-64]) * 100.0
                d10_3m = (merged["value_10y"].iloc[-1] - merged["value_10y"].iloc[-64]) * 100.0
                reg_3m = self.classify_yield_curve(d2_3m, d10_3m)
                horizon_labels["3M"] = reg_3m.label

            # Calculate 1Y frequency of current label
            if len(merged) >= 252:
                tail_1y = merged.tail(252).copy()
                tail_1y["d2"] = tail_1y["value_2y"].diff() * 100.0
                tail_1y["d10"] = tail_1y["value_10y"].diff() * 100.0
                labels_1y = [self.classify_yield_curve(r.d2, r.d10).label for _, r in tail_1y.dropna().iterrows()]
                match_cnt = sum(1 for l in labels_1y if l == label)
                frequency_1y = round((match_cnt / len(labels_1y)) * 100.0, 1)
                # Compute consecutive session duration
                dur = 0
                for l in reversed(labels_1y):
                    if l == label:
                        dur += 1
                    else:
                        break
                duration_sessions = max(1, dur)

        return YieldCurveRegime(
            label=label,
            plain_language_definition=defn,
            short_move_bp=delta_2y_bp,
            long_move_bp=delta_10y_bp,
            slope_change_bp=slope_delta,
            horizon_labels=horizon_labels,
            duration_sessions=duration_sessions,
            frequency_1y=frequency_1y,
            risk_implication=risk,
            evidence=evidence,
            status="OK",
        )

    def classify_cross_asset_regime(
        self, metrics: dict[str, CalculatedMetric], min_valid_components: int = 3
    ) -> RegimeClassification:
        """
        Transparent, multi-component cross-asset regime classification with aligned signed contributions.
        Missing components remain strictly missing and are not zero-filled.
        """
        scores: dict[str, float] = {}
        evidence: list[str] = []

        # 1. Equities: Positive return is Risk-On (+), negative is Risk-Off (-)
        sp = metrics.get("EQ_SP500")
        eq = metrics.get("EQ_EUROPE_600")
        eq_metric = sp if (sp and sp.change_value is not None) else (eq if (eq and eq.change_value is not None) else None)
        if eq_metric and eq_metric.change_value is not None:
            val = eq_metric.change_value
            s_eq = np.clip(val / 1.0, -3.0, 3.0)  # 1% move ~ 1 unit
            scores["Equities"] = float(round(s_eq, 2))
            evidence.append(f"Equities: {val:+.2f}% ({'Risk-On' if s_eq > 0 else 'Risk-Off'})")

        # 2. Credit Spreads: Widening (+ bp) is Risk-Off (-), Narrowing (- bp) is Risk-On (+)
        hy = metrics.get("US_HY_OAS")
        if hy and hy.change_value is not None:
            val = hy.change_value
            s_hy = np.clip(-val / 6.0, -3.0, 3.0)  # 6 bp widening ~ -1 unit
            scores["HY Credit OAS"] = float(round(s_hy, 2))
            evidence.append(f"US HY OAS: {val:+.1f} bp ({'Risk-On' if s_hy > 0 else 'Risk-Off'})")

        ig = metrics.get("US_IG_OAS")
        if ig and ig.change_value is not None:
            val = ig.change_value
            s_ig = np.clip(-val / 2.5, -3.0, 3.0)
            scores["IG Credit OAS"] = float(round(s_ig, 2))
            evidence.append(f"US IG OAS: {val:+.1f} bp ({'Risk-On' if s_ig > 0 else 'Risk-Off'})")

        # 3. Volatility (VIX): Rising VIX is Risk-Off (-), Falling VIX is Risk-On (+)
        vix = metrics.get("VIX_INDEX")
        if vix and vix.change_value is not None:
            val = vix.change_value
            s_vix = np.clip(-val / 5.0, -3.0, 3.0)  # 5% VIX spike ~ -1 unit
            scores["VIX Volatility"] = float(round(s_vix, 2))
            evidence.append(f"VIX Index: {val:+.1f}% ({'Risk-On' if s_vix > 0 else 'Risk-Off'})")

        # 4. FX EUR/USD
        fx = metrics.get("FX_EURUSD")
        if fx and fx.change_value is not None:
            val = fx.change_value
            s_fx = np.clip(val / 0.8, -2.0, 2.0)
            scores["EUR/USD"] = float(round(s_fx, 2))

        if len(scores) < min_valid_components:
            return RegimeClassification(
                label="Unclassified",
                plain_language_definition=f"Insufficient active market components ({len(scores)} valid of {min_valid_components} required).",
                score=None,
                confidence="LOW",
                evidence=evidence,
                status="INSUFFICIENT_COMPONENTS",
            )

        vals = list(scores.values())
        composite = float(np.mean(vals))
        dispersion = float(np.std(vals))

        # Check for conflicting divergent signals (dispersion high with significant polar components)
        has_strong_pos = any(v >= 0.8 for v in vals)
        has_strong_neg = any(v <= -0.8 for v in vals)

        if dispersion >= 1.0 and has_strong_pos and has_strong_neg:
            label = "Mixed / Divergent"
            defn = "Asset classes exhibit conflicting risk signals: equities and credit/volatility are decoupling."
        elif composite >= 1.25:
            label = "Strong Risk-On"
            defn = "Broad-based risk asset appreciation, contracting credit risk premiums, and falling volatility."
        elif composite >= 0.45:
            label = "Risk-On"
            defn = "Supportive macro risk appetite across equities and credit spreads."
        elif composite <= -1.25:
            label = "Acute Stress"
            defn = "Simultaneous equity liquidations, severe credit spread widening, and volatility spikes."
        elif composite <= -0.45:
            label = "Risk-Off"
            defn = "Defensive rotation: equity declines accompanied by widening credit spreads."
        else:
            label = "Neutral / Range-Bound"
            defn = "Subdued market movement without decisive cross-asset directional conviction."

        return RegimeClassification(
            label=label,
            plain_language_definition=defn,
            score=round(composite, 2),
            confidence="HIGH" if len(scores) >= 4 else "MEDIUM",
            evidence=evidence,
            component_contributions=scores,
            duration_sessions=1,
            status="OK",
        )