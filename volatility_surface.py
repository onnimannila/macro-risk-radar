"""BTC/USD market-implied volatility surface modeling and SVI calibration engine for Deribit."""

from datetime import datetime, timezone
import logging
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

from morning_report.config import DeribitVolatilitySurfaceConfig
from morning_report.models import BTCVolatilitySurfaceResult, OptionMarketObservation, SurfaceMode, SurfaceStatus

logger = logging.getLogger(__name__)


class RawSVICalibrator:
    """
    Fits Gatheral's Raw SVI parameterization per maturity slice:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where k = ln(K / F(T)), w(k) = total variance = sigma_implied^2 * T.
    """

    @staticmethod
    def svi_total_variance(k: np.ndarray | float, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray | float:
        disc = np.sqrt((k - m) ** 2 + sigma ** 2)
        return a + b * (rho * (k - m) + disc)

    @classmethod
    def fit_slice(
        cls,
        log_moneyness: np.ndarray,
        total_variance: np.ndarray,
        weights: np.ndarray | None = None,
        multistart_count: int = 12,
    ) -> dict[str, Any]:
        n = len(log_moneyness)
        if n < 5:
            raise ValueError(f"Insufficient observations for SVI slice: {n} quotes (minimum 5 required).")

        w_atm = float(np.interp(0.0, log_moneyness, total_variance)) if (log_moneyness.min() <= 0.0 <= log_moneyness.max()) else float(np.mean(total_variance))
        w_weights = weights if weights is not None else np.ones(n)
        w_weights = w_weights / np.sum(w_weights)

        def loss(p: list[float]) -> float:
            a, b, rho, m, sigma = p
            if b < 0.0 or sigma <= 1e-5 or abs(rho) >= 0.999:
                return 1e8
            # Minimum total variance constraint: a + b*sigma*sqrt(1 - rho^2) >= 0
            min_val = a + b * sigma * math.sqrt(max(1e-6, 1.0 - rho * rho))
            if min_val < 0.0:
                return 1e8 + abs(min_val) * 1e5
            model_w = cls.svi_total_variance(log_moneyness, a, b, rho, m, sigma)
            res = (model_w - total_variance)
            return float(np.sum(w_weights * (res ** 2)))

        bounds = [
            (-1.0, max(2.0, w_atm * 2.0)),      # a
            (1e-4, 2.5),                          # b
            (-0.95, 0.95),                        # rho
            (-0.5, 0.5),                          # m
            (1e-3, 1.0),                          # sigma
        ]

        # Deterministic multi-start seeds across common skew and smile shapes
        rng = np.random.default_rng(20261111)
        x0_candidates = [
            [w_atm * 0.5, 0.15, -0.20, 0.0, 0.10],
            [w_atm * 0.8, 0.08, -0.40, 0.02, 0.15],
            [w_atm * 0.3, 0.25, 0.10, -0.02, 0.08],
            [w_atm * 0.6, 0.18, -0.10, 0.0, 0.20],
        ]
        for _ in range(max(1, multistart_count - len(x0_candidates))):
            x0_candidates.append([
                float(rng.uniform(w_atm * 0.2, w_atm * 0.9)),
                float(rng.uniform(0.02, 0.40)),
                float(rng.uniform(-0.60, 0.20)),
                float(rng.uniform(-0.15, 0.15)),
                float(rng.uniform(0.03, 0.30)),
            ])

        best_loss = float("inf")
        best_params = x0_candidates[0]
        converged = False

        for x0 in x0_candidates:
            try:
                opt = minimize(loss, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
                if opt.fun < best_loss:
                    best_loss = opt.fun
                    best_params = [float(val) for val in opt.x]
                    converged = opt.success
            except Exception:
                continue

        a, b, rho, m, sigma = best_params
        fit_w = cls.svi_total_variance(log_moneyness, a, b, rho, m, sigma)
        residuals = fit_w - total_variance
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        return {
            "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
            "rmse": rmse, "loss": best_loss, "converged": converged,
            "residuals": residuals.tolist(),
        }


class BTCVolatilitySurfaceEngine:
    """
    Production calibrator for Deribit BTC option market volatility surfaces.
    Pairs liquid OTM calls and puts, fits per-expiry Raw SVI smiles, performs monotone
    PCHIP total-variance term structure interpolation, and evaluates dense 3D and 2D matrices.
    """

    def __init__(self, config: DeribitVolatilitySurfaceConfig | None = None) -> None:
        self.config = config or DeribitVolatilitySurfaceConfig()

    def build_btc_surface(
        self,
        observations: list[OptionMarketObservation],
        execution_time: datetime | None = None,
    ) -> BTCVolatilitySurfaceResult:
        val_time = execution_time or datetime.now(timezone.utc)
        helsinki_str = val_time.astimezone().strftime("%Y-%m-%d %H:%M:%S")

        if not observations:
            return BTCVolatilitySurfaceResult(
                surface_type=SurfaceMode.UNAVAILABLE,
                provider="DERIBIT",
                underlying="BTC",
                valuation_timestamp=val_time,
                valuation_timestamp_helsinki=helsinki_str,
                status=SurfaceStatus.UNAVAILABLE,
                warnings=["No observations provided to surface engine."],
                errors=["Deribit BTC options observation list is empty."],
            )

        # Count exclusions
        exclusions_by_reason: dict[str, int] = {}
        valid_obs: list[OptionMarketObservation] = []
        spot_ref = 0.0
        for obs in observations:
            if obs.underlying_price > spot_ref:
                spot_ref = obs.underlying_price
            if obs.quality_status == "VALID":
                valid_obs.append(obs)
            else:
                r = obs.exclusion_reason or "Unknown exclusion"
                exclusions_by_reason[r] = exclusions_by_reason.get(r, 0) + 1

        if len(valid_obs) < self.config.minimum_total_quotes:
            return BTCVolatilitySurfaceResult(
                surface_type=SurfaceMode.UNAVAILABLE,
                provider="DERIBIT",
                underlying="BTC",
                valuation_timestamp=val_time,
                valuation_timestamp_helsinki=helsinki_str,
                spot_or_index_price=spot_ref,
                raw_instrument_count=len(observations),
                valid_observation_count=len(valid_obs),
                excluded_quote_count=len(observations) - len(valid_obs),
                exclusions_by_reason=exclusions_by_reason,
                status=SurfaceStatus.UNAVAILABLE,
                warnings=[f"Insufficient valid quotes ({len(valid_obs)} < {self.config.minimum_total_quotes})."],
            )

        # Group by expiry date
        by_expiry: dict[str, list[OptionMarketObservation]] = {}
        for obs in valid_obs:
            exp_key = obs.expiry_timestamp.strftime("%Y-%m-%d")
            by_expiry.setdefault(exp_key, []).append(obs)

        # Select OTM contracts and filter expiries with >= minimum_quotes_per_expiry
        retained_expiries: list[str] = []
        retained_maturities: list[float] = []
        retained_quotes: list[OptionMarketObservation] = []
        svi_by_expiry: dict[str, dict[str, Any]] = {}
        rmse_by_expiry: dict[str, float] = {}
        call_put_discrepancy_count = 0
        discrepancy_threshold = self.config.call_put_discrepancy_threshold_pts / 100.0

        sorted_expiries = sorted(by_expiry.keys(), key=lambda k: by_expiry[k][0].maturity_years)

        for exp_key in sorted_expiries:
            chain = by_expiry[exp_key]
            # Group by strike
            by_strike: dict[float, dict[str, OptionMarketObservation]] = {}
            for c in chain:
                by_strike.setdefault(c.strike, {})[c.option_type] = c

            chosen_for_expiry: list[OptionMarketObservation] = []
            for strike_val, pair in by_strike.items():
                c_opt = pair.get("call")
                p_opt = pair.get("put")

                # Check put-call discrepancy
                if c_opt and p_opt and c_opt.mark_iv and p_opt.mark_iv:
                    diff = abs(c_opt.mark_iv - p_opt.mark_iv)
                    if diff > discrepancy_threshold:
                        call_put_discrepancy_count += 1

                # Prefer OTM: Put if K < F, Call if K >= F
                fwd_ref = (c_opt.forward_price if c_opt else (p_opt.forward_price if p_opt else spot_ref))
                if strike_val < fwd_ref:
                    chosen = p_opt or c_opt
                else:
                    chosen = c_opt or p_opt

                if chosen and chosen.mark_iv is not None:
                    chosen_for_expiry.append(chosen)

            if len(chosen_for_expiry) < self.config.minimum_quotes_per_expiry:
                continue

            chosen_for_expiry.sort(key=lambda o: o.strike)
            tau = chosen_for_expiry[0].maturity_years
            k_arr = np.array([o.log_forward_moneyness for o in chosen_for_expiry])
            iv_arr = np.array([o.mark_iv for o in chosen_for_expiry])
            tot_var = (iv_arr ** 2) * tau

            # Liquidity weights: square-root of open interest + volume
            weights = np.array([math.sqrt(max(0.1, o.open_interest + o.volume * 0.5)) for o in chosen_for_expiry])
            weights = np.clip(weights, 0.2, 5.0)

            try:
                fit_res = RawSVICalibrator.fit_slice(
                    k_arr, tot_var, weights=weights, multistart_count=self.config.multistart_count
                )
                retained_expiries.append(exp_key)
                retained_maturities.append(tau)
                retained_quotes.extend(chosen_for_expiry)
                svi_by_expiry[exp_key] = fit_res
                # Store RMSE in volatility points (%)
                rmse_iv = (fit_res["rmse"] / (2.0 * math.sqrt(max(1e-4, tau * np.mean(tot_var))))) * 100.0 if np.mean(tot_var) > 0 else fit_res["rmse"] * 100.0
                rmse_by_expiry[exp_key] = float(round(rmse_iv, 2))
            except Exception as exc:
                logger.warning("SVI calibration failed for expiry %s: %s", exp_key, exc)
                continue

        if len(retained_expiries) < self.config.minimum_expiries or len(retained_quotes) < self.config.minimum_total_quotes:
            status = SurfaceStatus.PARTIAL if len(retained_expiries) >= 2 else SurfaceStatus.UNAVAILABLE
            return BTCVolatilitySurfaceResult(
                surface_type=SurfaceMode.MARKET_IMPLIED,
                provider="DERIBIT",
                underlying="BTC",
                valuation_timestamp=val_time,
                valuation_timestamp_helsinki=helsinki_str,
                spot_or_index_price=spot_ref,
                raw_instrument_count=len(observations),
                valid_observation_count=len(valid_obs),
                retained_expiries=retained_expiries,
                retained_quote_count=len(retained_quotes),
                excluded_quote_count=len(observations) - len(retained_quotes),
                exclusions_by_reason=exclusions_by_reason,
                status=status,
                warnings=[f"Retained {len(retained_expiries)} expiries and {len(retained_quotes)} quotes (thresholds: {self.config.minimum_expiries} expiries, {self.config.minimum_total_quotes} quotes)."],
            )

        # Evaluate Dense Grid across continuous maturities and log-forward moneyness
        dense_n_t = self.config.dense_maturity_points
        dense_n_k = self.config.dense_moneyness_points
        min_tau, max_tau = min(retained_maturities), max(retained_maturities)
        # Short-end weighted maturity grid
        t_norm = np.linspace(0.0, 1.0, dense_n_t)
        dense_mats = (min_tau + (max_tau - min_tau) * (t_norm ** 1.35)).tolist()

        # Common supported log-moneyness range derived from observations
        min_k_obs = max(-0.45, float(min(o.log_forward_moneyness for o in retained_quotes)))
        max_k_obs = min(0.45, float(max(o.log_forward_moneyness for o in retained_quotes)))
        dense_k = np.linspace(min_k_obs, max_k_obs, dense_n_k).tolist()
        dense_fwd_m = [float(math.exp(k)) for k in dense_k]

        # Compute SVI total variance for each retained expiry slice across the moneyness grid
        retained_slice_w = np.zeros((len(retained_expiries), dense_n_k))
        for i, exp_k in enumerate(retained_expiries):
            p = svi_by_expiry[exp_k]
            retained_slice_w[i, :] = RawSVICalibrator.svi_total_variance(
                np.array(dense_k), p["a"], p["b"], p["rho"], p["m"], p["sigma"]
            )

        # Monotone PCHIP cubic spline interpolation in maturity for each moneyness column
        dense_vol_matrix: list[list[float | None]] = []
        calendar_violations = 0
        negative_variance_count = 0

        for t_dense in dense_mats:
            row_dense: list[float | None] = []
            for j in range(dense_n_k):
                w_col = retained_slice_w[:, j]
                # Monotone cubic interpolation prevents calendar crossings across maturity
                interpolator = PchipInterpolator(retained_maturities, w_col)
                w_val = float(interpolator(t_dense))
                if w_val <= 0:
                    negative_variance_count += 1
                    row_dense.append(None)
                else:
                    iv = math.sqrt(w_val / t_dense)
                    row_dense.append(float(round(iv, 5)))
            dense_vol_matrix.append(row_dense)

        # Calendar total variance monotonicity check
        for i in range(1, dense_n_t):
            for j in range(dense_n_k):
                v_prev = dense_vol_matrix[i - 1][j]
                v_curr = dense_vol_matrix[i][j]
                if v_prev is not None and v_curr is not None:
                    w_prev = (v_prev ** 2) * dense_mats[i - 1]
                    w_curr = (v_curr ** 2) * dense_mats[i]
                    if w_curr < w_prev - 1e-5:
                        calendar_violations += 1

        # SVI Butterfly Convexity Diagnostic (Gatheral g(k) check)
        butterfly_violations = 0
        for exp_k in retained_expiries:
            p = svi_by_expiry[exp_k]
            a, b, rho, m, sigma = p["a"], p["b"], p["rho"], p["m"], p["sigma"]
            # Check g(k) >= 0 across dense_k
            for k in dense_k:
                y = k - m
                disc = math.sqrt(y * y + sigma * sigma)
                w_val = a + b * (rho * y + disc)
                w_prime = b * (rho + y / disc)
                w_prime2 = b * (sigma * sigma) / (disc ** 3)
                g_k = (1.0 - k * w_prime / (2.0 * max(1e-6, w_val))) ** 2 - (w_prime ** 2) / 4.0 * (1.0 / max(1e-6, w_val) + 0.25) + w_prime2 / 2.0
                if g_k < -0.01:
                    butterfly_violations += 1

        total_dense_points = dense_n_t * dense_n_k
        successful_pts = sum(1 for row in dense_vol_matrix for v in row if v is not None and v > 0)
        coverage_pct = (successful_pts / total_dense_points * 100.0) if total_dense_points > 0 else 0.0

        calls_ret = sum(1 for o in retained_quotes if o.option_type == "call")
        puts_ret = sum(1 for o in retained_quotes if o.option_type == "put")
        avg_oi = float(np.mean([o.open_interest for o in retained_quotes]))
        avg_spread = float(np.mean([abs((o.ask_iv or o.mark_iv or 0) - (o.bid_iv or o.mark_iv or 0)) * 100.0 for o in retained_quotes]))
        weighted_rmse = float(np.mean(list(rmse_by_expiry.values()))) if rmse_by_expiry else 0.0

        status = SurfaceStatus.AVAILABLE
        if coverage_pct < 98.0 or calendar_violations > 20:
            status = SurfaceStatus.AVAILABLE_WITH_WARNINGS
        if len(retained_expiries) < 4 or len(retained_quotes) < 80:
            status = SurfaceStatus.PARTIAL

        warnings = []
        if calendar_violations > 0:
            warnings.append(f"Calendar total variance monotonicity non-compliance at {calendar_violations} grid points.")
        if butterfly_violations > 0:
            warnings.append(f"SVI butterfly convexity non-compliance at {butterfly_violations} evaluated moneyness points.")
        if call_put_discrepancy_count > 0:
            warnings.append(f"Detected {call_put_discrepancy_count} strike pairs with Call/Put IV discrepancy > {self.config.call_put_discrepancy_threshold_pts:.1f} vol pts.")

        return BTCVolatilitySurfaceResult(
            surface_type=SurfaceMode.MARKET_IMPLIED,
            provider="DERIBIT",
            underlying="BTC",
            quote_currency="USD",
            valuation_timestamp=val_time,
            valuation_timestamp_helsinki=helsinki_str,
            spot_or_index_price=spot_ref,
            raw_instrument_count=len(observations),
            valid_observation_count=len(valid_obs),
            retained_expiries=retained_expiries,
            retained_expiry_maturities=retained_maturities,
            retained_quote_count=len(retained_quotes),
            excluded_quote_count=len(observations) - len(retained_quotes),
            exclusions_by_reason=exclusions_by_reason,
            calls_retained=calls_ret,
            puts_retained=puts_ret,
            average_open_interest=round(avg_oi, 1),
            average_bid_ask_spread_iv=round(avg_spread, 2),
            observed_expiries_years=[o.maturity_years for o in retained_quotes],
            observed_log_moneyness=[o.log_forward_moneyness for o in retained_quotes],
            observed_forward_moneyness=[o.forward_moneyness for o in retained_quotes],
            observed_volatilities_pct=[(o.mark_iv or 0.0) * 100.0 for o in retained_quotes],
            observed_option_types=[o.option_type for o in retained_quotes],
            dense_maturity_grid=dense_mats,
            dense_moneyness_grid=dense_k,
            dense_forward_moneyness_grid=dense_fwd_m,
            dense_vol_matrix=dense_vol_matrix,
            calibration_method="Quasi-Explicit Raw SVI with Monotone Spline Total-Variance Term Structure",
            calibration_parameters=svi_by_expiry,
            calibration_rmse_by_expiry=rmse_by_expiry,
            weighted_rmse=round(weighted_rmse, 2),
            direct_observation_count=len(retained_quotes),
            dense_model_point_count=total_dense_points,
            dense_grid_coverage_pct=round(coverage_pct, 1),
            arbitrage_diagnostics={
                "calendar_arbitrage_violations": calendar_violations,
                "calendar_status": "PASSED" if calendar_violations == 0 else "WARNING",
                "butterfly_arbitrage_violations": butterfly_violations,
                "butterfly_status": "PASSED" if butterfly_violations == 0 else "WARNING",
                "negative_total_variance_count": negative_variance_count,
            },
            liquidity_diagnostics={
                "call_put_discrepancy_count": call_put_discrepancy_count,
                "discrepancy_threshold_pts": self.config.call_put_discrepancy_threshold_pts,
                "average_open_interest": round(avg_oi, 1),
                "average_bid_ask_iv_spread_pct": round(avg_spread, 2),
            },
            warnings=warnings,
            status=status,
        )