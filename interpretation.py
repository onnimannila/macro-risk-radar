"""Deterministic, data-reactive market interpretations and portfolio risk commentary."""

from datetime import date
import json
import logging
import os
from pathlib import Path
import re
from morning_report.models import BTCVolatilitySurfaceResult, CalculatedMetric, RegimeClassification, YieldCurveRegime
import numpy as np

logger = logging.getLogger(__name__)


class HeadlineEngine:
    """
    Data-reactive market headline generator with 10-calendar-day uniqueness enforcement,
    local persistent JSON storage, atomic saves, and same-day rerun stability.
    """

    THEME_CANDIDATES: dict[str, list[dict[str, str]]] = {
        "STRONG_RISK_ON": [
            {"text": "Risk-On Has Entered the Chat: Equities Lead the Surge", "family": "entered_chat"},
            {"text": "Bulls Take the Floor: Equities Rally as Volatility Retreats", "family": "bulls_rally"},
            {"text": "Green Screens Across the Board: Credit Compresses and Stocks Climb", "family": "green_screens"},
            {"text": "Equities Charge Ahead While Credit Spreads Tighten Fast", "family": "charge_ahead"},
            {"text": "Risk Appetite Uncorked: Portfolios Lean Firmly into the Rally", "family": "appetite_uncorked"},
            {"text": "Volatility Tumbles as Global Equity Allocators Press the Gas", "family": "vol_tumbles"},
            {"text": "Credit Spreads Shrink as Equity Momentum Takes Full Control", "family": "spreads_shrink"},
            {"text": "Spreads Tighten, Stocks Soar, and Risk Budgets Expand", "family": "spreads_stocks_soar"},
            {"text": "Equities Bring the Confetti, Bonds Read the Fine Print", "family": "confetti_fine_print"},
            {"text": "Green Lights Across Trading Desks as Animal Spirits Roar Back", "family": "green_lights"},
            {"text": "Bulls Throw a Block Party While Volatility Packs Its Bags", "family": "block_party"},
            {"text": "Risk Budgets Open Wide: Equities Fly and Spreads Vanish", "family": "budgets_open"},
            {"text": "All Aboard the Risk Express: Equities Surge Past Hesitant Bears", "family": "risk_express"},
            {"text": "Full Throttle: Equity Bids Multiply While Credit Fear Dissolves", "family": "full_throttle"},
            {"text": "Euphoria in the Front Seat: Equities Lead High-Beta Assets Skyward", "family": "euphoria_front"},
            {"text": "Risk Appetite in Overdrive: Equities Power Past Resistance", "family": "appetite_overdrive"},
            {"text": "High Beta Celebration: Stocks Leap as Risk Premiums Disappear", "family": "beta_celebration"},
            {"text": "Bulls Break the Tape: Unrelenting Momentum Across Global Equities", "family": "break_tape"},
            {"text": "Party on the Desk: Equity Allocators Push Risk to the Max", "family": "party_desk"},
            {"text": "Green Waves Everywhere: Equities Vault Higher with Credit in Tow", "family": "green_waves"},
        ],
        "RISK_ON": [
            {"text": "Bonds Breathe Out, Equities Order Another Espresso", "family": "espresso_analogy"},
            {"text": "Equities Grind Higher While Corporate Credit Holds the Line", "family": "grind_higher"},
            {"text": "Supportive Risk Currents: Stocks Advance and Volatility Simmers Down", "family": "supportive_currents"},
            {"text": "Bulls Stay in Office as Risk Appetite Extends Its Reach", "family": "bulls_office"},
            {"text": "Calmer Waters: Equities Advance While Spreads Behave", "family": "calmer_waters"},
            {"text": "Risk Assets Keep Smiling: Modest Gains and Muted Dispersion", "family": "keep_smiling"},
            {"text": "Carry Trumps Caution: Equities Drift Upward in Orderly Trade", "family": "carry_trumps"},
            {"text": "Steady Risk Appetite: Equities Edge Ahead with Credit Calm", "family": "steady_appetite"},
            {"text": "Equities Strut Their Stuff While Corporate Credit Applauds", "family": "strut_stuff"},
            {"text": "Gentle Tailwinds: Stocks March Upward as Volatility Drifts Asleep", "family": "gentle_tailwinds"},
            {"text": "Risk Appetite on Cruise Control: Orderly Buying Across Equities", "family": "cruise_control"},
            {"text": "Constructive Tape: Equities Creep Higher While Spreads Behave", "family": "constructive_tape"},
            {"text": "Bulls Enjoy the Sunshine: Smooth Sailing Across Core Risk Sleeves", "family": "enjoy_sunshine"},
            {"text": "Credit Spreads Hum Peacefully as Equity Buyers Nibble", "family": "hum_peacefully"},
            {"text": "Quiet Confidence: Risk Portfolios Pocket Another Session of Gains", "family": "quiet_confidence"},
            {"text": "Equities Sip Champagne as Volatility Keeps Its Distance", "family": "sip_champagne"},
            {"text": "Brisk Spring Step: Stocks Float Higher on Benign Credit Tape", "family": "spring_step"},
            {"text": "Sunny Outlook on the Desk: Risk Bids Percolate Without Drama", "family": "sunny_outlook"},
            {"text": "Orderly Optimism: Equities Step Forward as Carry Trades Pay Off", "family": "orderly_optimism"},
            {"text": "Equities Lead with a Grin While Fixed Income Keeps Calm", "family": "lead_with_grin"},
        ],
        "ACUTE_STRESS": [
            {"text": "Spreads Blow Out, Equities Drop, and VIX Demands Attention", "family": "blow_out"},
            {"text": "Red Tape Across Desks: Acute Risk-Off Shakes Allocations", "family": "red_tape"},
            {"text": "Defensive Helmets Fastened: Credit Widens Sharply as Equities Sink", "family": "helmets_on"},
            {"text": "Volatility Erupts: Liquidity Retreats and Risk Budgets Retract", "family": "vol_erupts"},
            {"text": "Spreads Check the Fire Exits as Equities Take Heavy Fire", "family": "checks_exits"},
            {"text": "Severe Risk Aversion Takes the Wheel Across Global Asset Classes", "family": "severe_aversion"},
            {"text": "Duration Scrambles for Shelter as Equity Bids Disappear", "family": "duration_shelter"},
            {"text": "Sirens on the Trading Floor: Credit Cracks and Equities Capitulate", "family": "floor_sirens"},
            {"text": "Sell-Off Gathers Speed: Spreads Fracture as VIX Jumps the Fence", "family": "gathers_speed"},
            {"text": "Bunker Mentality: Liquidity Dries Up as Allocators Scramble for Safety", "family": "bunker_mentality"},
            {"text": "Turbulence Alert: Equity Bids Evaporate Amid Acute Risk Flight", "family": "turbulence_alert"},
            {"text": "Risk Budgets on Life Support: Broad Liquidations Hit Global Markets", "family": "life_support"},
            {"text": "Flashpoints Multiply: Credit Deterioration Triggers Swift Defensive Unwind", "family": "flashpoints_mult"},
            {"text": "Alarm Bells on the Desk: Risk Assets Liquefied in Flight to Cash", "family": "alarm_bells"},
            {"text": "Storm Clouds Break Overhead: High Beta Crumbles Under Liquidation", "family": "storm_clouds"},
            {"text": "Panic at the Gates: Volatility Surges as Protective Bids Dry Up", "family": "panic_gates"},
            {"text": "Brace for Impact: Severe Drawdowns Sweep Multi-Asset Sleeves", "family": "brace_impact"},
            {"text": "Spreads Gape Wide Open as Equity Portfolios Hit Emergency Brakes", "family": "spreads_gape"},
            {"text": "Red Ink Uncontained: Global Liquidations Sweep Trading Desks", "family": "red_ink"},
            {"text": "Defensive Armor Pierced: Acute Selling Hammers Equities and Credit", "family": "armor_pierced"},
        ],
        "RISK_OFF": [
            {"text": "Markets Smile Wanly, Credit Checks the Exits", "family": "checks_exits"},
            {"text": "Defensive Rotation: Equities Step Back as Credit Spreads Flinch", "family": "defensive_rot"},
            {"text": "Equities Trim Sails While Corporate Spreads Creep Wider", "family": "trim_sails"},
            {"text": "Caution Takes the Aux Cord: Risk Assets Hand Back Gains", "family": "aux_cord"},
            {"text": "Risk Budgets Tighten as Equity Momentum Hits a Wall", "family": "hits_wall"},
            {"text": "Duration Finds Friends as Equities Drift into the Red", "family": "finds_friends"},
            {"text": "Credit Spreads Widen and Equity Allocators Reach for Hedges", "family": "reaches_hedges"},
            {"text": "Bulls Take an Early Lunch as Defensive Assets Catch the Bid", "family": "early_lunch"},
            {"text": "Risk Appetite Retreats Indoors as Market Clouds Gather", "family": "retreats_indoors"},
            {"text": "Equities Lose Their Footing While Safe Havens Find Buyers", "family": "lose_footing"},
            {"text": "Sober Morning: Allocators Pare Back Risk as Spreads Soften", "family": "sober_morning"},
            {"text": "Caution in the Air: Stocks Ease Back and Credit Demands a Premium", "family": "caution_air"},
            {"text": "Defensive Playbook Opens: Portfolio Managers Reach for Quality", "family": "defensive_playbook"},
            {"text": "Brisk Risk-Off Drift: Volatility Ticks Up as Equity Buyers Step Aside", "family": "risk_off_drift"},
            {"text": "Risk Assets Tap the Brakes as Caution Creeps Over Desks", "family": "tap_brakes"},
            {"text": "Pulling Back Bids: Equities Soften While Havens Catch Attention", "family": "pulling_back"},
            {"text": "Umbrellas Unfurled: Allocators Seek Shelter as Momentum Fades", "family": "umbrellas_unfurled"},
            {"text": "Hedging Desks Clock In: Stocks Ease as Credit Demands Better Terms", "family": "hedging_clock_in"},
            {"text": "Aversion Takes the Lead: Modest Losses Prompt Defensive Positioning", "family": "aversion_lead"},
            {"text": "Trading Desks Don Hard Hats as Equity Bids Thin Out", "family": "hard_hats"},
        ],
        "MIXED_DIVERGENT": [
            {"text": "Cross-Asset Decoupling: Equities Whistle Past Credit's Wary Glances", "family": "decoupling_whistle"},
            {"text": "Bipolar Tape: Equities March While Credit Taps the Brakes", "family": "bipolar_tape"},
            {"text": "Cross-Asset Tug of War: Equities and Spreads Part Ways", "family": "tug_of_war"},
            {"text": "Divergent Currents: Stock Optimism Meets Credit Skepticism", "family": "divergent_currents"},
            {"text": "Conflicting Signals on the Desk: Risk Assets Disagree on Direction", "family": "conflicting_signals"},
            {"text": "Equities Push Forward While Corporate Spreads Refuse to Follow", "family": "refuse_follow"},
            {"text": "A Tale of Two Sleeves: Stocks Advance as Spreads Hesitate", "family": "two_sleeves"},
            {"text": "Cross-Asset Disconnect: Stock Euphoria Clashes with Bond Sobriety", "family": "disconnect_clashes"},
            {"text": "Schrödinger’s Market: Risk Assets Lean Both Ways at Once", "family": "schrodinger_market"},
            {"text": "Equities Take the High Road, Credit Takes the Scenic Detour", "family": "high_road"},
            {"text": "Asset Classes Speak in Different Tongues: Conflicting Directional Prints", "family": "different_tongues"},
            {"text": "Muddled Macro Messages: Allocators Juggle Decoupled Risk Drivers", "family": "muddled_messages"},
            {"text": "Uneven Footing: High Beta Rallies While Fixed Income Shrugs", "family": "uneven_footing"},
            {"text": "Two Different Movies: Stocks Party While Credit Sits Out", "family": "two_movies"},
            {"text": "Cross-Currents on the Tape: Conflicting Pulses Puzzle Allocators", "family": "cross_currents"},
            {"text": "Decoupled Desks: Equities and Credit Cannot Agree on Direction", "family": "decoupled_desks"},
            {"text": "Mixed Signals Everywhere: Risk Markets Split Down the Middle", "family": "mixed_everywhere"},
            {"text": "Equities Wave Green Flags While Fixed Income Flashes Yellow", "family": "flags_clash"},
            {"text": "Contradictory Currents: Momentum and Spreads Go Their Separate Ways", "family": "currents_separate"},
            {"text": "Asymmetric Tape: Pockets of Cheer Clash with Pockets of Caution", "family": "asymmetric_tape"},
        ],
        "RATES_BEAR_MOVE": [
            {"text": "Sovereign Yields Step Up: Liability Relief Meets Duration Headwinds", "family": "yields_step_up"},
            {"text": "Rates Take the Spotlight: Benchmark Yields Surge Across Maturities", "family": "rates_spotlight"},
            {"text": "Duration Feels the Pinch as Core European Yields March Higher", "family": "duration_pinch"},
            {"text": "Curve Bears on the Prowl: Rising Yields Reprice Fixed Income", "family": "curve_bears"},
            {"text": "Rates Drive the Morning: Discount Rates Jump on Sovereign Selling", "family": "rates_drive"},
            {"text": "Yields Turn Up the Heat: Duration Portfolios Feel the Sunburn", "family": "turn_up_heat"},
            {"text": "Bond Desks Eye the Exits as Sovereign Benchmarks Push Higher", "family": "bond_desks_exits"},
            {"text": "Discount Rates Flex Their Muscles Across Global Sovereign Curves", "family": "rates_flex"},
            {"text": "Duration Stumbles: Core European Yields Reassert Upward Pressure", "family": "duration_stumbles"},
            {"text": "Fixed Income Repricing: Long-End Yields Climb on Term Premium Jitters", "family": "fixed_income_repricing"},
            {"text": "Yield Curve Steps Higher: Benchmark Sovereigns Reprice Discount Rates", "family": "curve_steps_higher"},
            {"text": "Term Premiums Demand Payment: Benchmark Yields Push Upward", "family": "term_premiums"},
            {"text": "Duration Headwinds Stiffen as Sovereign Sellers Take Control", "family": "headwinds_stiffen"},
            {"text": "Discount Rates Creep North: Fixed Income Portfolios Adjust Expectations", "family": "rates_creep_north"},
            {"text": "Sovereign Paper Under Pressure: Yields Climb Across Benchmark Tenors", "family": "paper_pressure"},
            {"text": "Rates Pressure Builds: Rising Yields Reset Multi-Asset Valuations", "family": "rates_pressure_builds"},
            {"text": "Curve Bears Dictate Terms: Discount Rates Jump on Fixed Income Benches", "family": "bears_dictate"},
            {"text": "Duration Feels the Weight as Sovereign Yields Move Higher", "family": "feels_weight"},
            {"text": "Fixed Income Takes a Bruise: Core Curves Reprice Discount Pressures", "family": "takes_bruise"},
            {"text": "Bond Desks Clear Room as Sovereign Benchmark Yields Climb", "family": "clear_room"},
        ],
        "RATES_BULL_MOVE": [
            {"text": "Sovereign Curves Rally: Duration Enjoys a Welcome Morning Lift", "family": "curves_rally"},
            {"text": "Yields Retreat Across Tenors: Fixed Income Sleeves Catch a Bid", "family": "yields_retreat"},
            {"text": "Bonds Stage a Comeback: Benchmark Discount Rates Drift Lower", "family": "bonds_comeback"},
            {"text": "Fixed Income Sighs in Relief as Core Sovereign Yields Drop", "family": "sighs_relief"},
            {"text": "Duration Wins the Session: Falling Yields Cushion Portfolio Valuations", "family": "duration_wins"},
            {"text": "Bonds Catch Fresh Wind: Core Sovereign Yields Ease Across Tenors", "family": "bonds_fresh_wind"},
            {"text": "Sovereign Bulls Step In: Benchmark Yields Retreat from Recent Highs", "family": "bulls_step_in"},
            {"text": "Fixed Income Hugs Duration as Benchmark Yields Drift Lower", "family": "hugs_duration"},
            {"text": "Rates Ease Their Grip: Bond Valuations Rebound on Sovereign Buying", "family": "ease_grip"},
            {"text": "Welcome Tailwinds for Duration: Sovereign Curves Rally in Harmony", "family": "welcome_tailwinds"},
            {"text": "Yields Slip Lower: Fixed Income Portfolios Find Calm Seas", "family": "yields_slip"},
            {"text": "Duration Breathes Easy: Core Sovereign Benchmarks Rally Across Tenors", "family": "breathes_easy"},
            {"text": "Green Shoots for Bonds: Yields Ease Across European and US Curves", "family": "green_shoots"},
            {"text": "Fixed Income Catches a Respite: Benchmark Yields Push Downward", "family": "catches_respite"},
            {"text": "Sovereign Paper in High Demand: Falling Yields Buoy Bond Sleeves", "family": "high_demand"},
            {"text": "Discount Rates Back Off: Duration Portfolios Welcome Yield Relief", "family": "back_off"},
            {"text": "Rally Along the Curve: Fixed Income Portfolios Collect Morning Gains", "family": "rally_along"},
            {"text": "Yields Soften Gently: Sovereign Curves Provide Welcome Valuation Support", "family": "soften_gently"},
            {"text": "Duration Takes the Crown: Sovereign Buyers Push Discount Rates Down", "family": "takes_crown"},
            {"text": "Calm Waters for Fixed Income: Benchmark Yields Retreat in Good Order", "family": "calm_waters"},
        ],
        "RANGE_BOUND": [
            {"text": "Range-Bound Tape: Macro Indicators Keep Their Secrets Today", "family": "keep_secrets"},
            {"text": "Sideways Morning: Asset Allocators Wait for the Next Catalyst", "family": "wait_catalyst"},
            {"text": "Subdued Drift: Markets Tiptoe into the New Trading Session", "family": "subdued_drift"},
            {"text": "Quiet Consensus: Cross-Asset Indicators Hold Their Morning Ground", "family": "quiet_consensus"},
            {"text": "Low-Beta Coffee Break: Asset Classes Trade in Quiet Balance", "family": "coffee_break"},
            {"text": "Holding the Line: Subdued Volatility Keeps Portfolios Range-Bound", "family": "holding_line"},
            {"text": "Waiting for Fresh Impulses: Macro Signals Hold Steady", "family": "fresh_impulses"},
            {"text": "Markets Hit Snooze, Volatility Keeps One Eye Open", "family": "hit_snooze"},
            {"text": "Flatlining Ticker Tape: Allocators Wait for Someone to Make a Move", "family": "flatlining_tape"},
            {"text": "Macro in Low Gear: Directional Momentum Takes the Morning Off", "family": "macro_low_gear"},
            {"text": "Sideways Drift: Cross-Asset Classes Trade in Muted Equilibrium", "family": "sideways_drift"},
            {"text": "Quiet Across the Board: Asset Allocators Sip Tea and Watch the Tape", "family": "sip_tea"},
            {"text": "Patience on the Desk: Macro Signals Tread Familiar Water", "family": "patience_desk"},
            {"text": "Equilibrium Rules the Morning: Risk Assets Drift in Search of a Driver", "family": "equilibrium_rules"},
            {"text": "Quiet Tape Unfolds: Portfolios Mark Time in Muted Trading", "family": "tape_unfolds"},
            {"text": "Subdued Session: Asset Allocators Sit Tight and Watch the Clock", "family": "subdued_session"},
            {"text": "Neutral Currents: Cross-Asset Indicators Hover in Equilibrium", "family": "neutral_currents"},
            {"text": "Tape on Standby: Markets Await Clear Directional Catalysts", "family": "standby_tape"},
            {"text": "Steady As She Goes: Subdued Moves Keep Risk Budgets Centered", "family": "steady_she_goes"},
            {"text": "Equilibrium Prevails: Calm Session Leaves Portfolio Hedges Inactive", "family": "equilibrium_prevails"},
        ],
        "INCONCLUSIVE_DATA": [
            {"text": "Data Fog Descends: Allocators Await Clearer Signal Prints", "family": "data_fog"},
            {"text": "Inconclusive Tape: Macro Indicators Keep Their Powder Dry", "family": "powder_dry"},
            {"text": "Partial Feeds and Quiet Tape: Risk Monitors Maintain Holding Patterns", "family": "partial_feeds"},
            {"text": "Muted Clarity: Risk Engines Stand By for Fresh Confirmation Prints", "family": "muted_clarity"},
            {"text": "Waiting on Foggy Feeds: Risk Monitors Hold Steady for Clearer Signals", "family": "foggy_feeds"},
            {"text": "Sparse Prints on the Desk: Market Direction Stays Behind Closed Doors", "family": "sparse_prints"},
            {"text": "Data Feeds Play Hard to Get: Asset Monitors Stand in Active Vigilance", "family": "hard_to_get"},
            {"text": "Unfinished Sentences: Macro Signals Await Confirmation from the Next Session", "family": "unfinished_sentences"},
            {"text": "Signal Fog Over Trading Desks: Risk Engines Maintain Steady Stance", "family": "signal_fog"},
            {"text": "Radar Scanning in the Mist: Macro Feeds Offer Muted Clarity Today", "family": "radar_scanning"},
            {"text": "Incomplete Pictures: Risk Systems Hold Ground While Feeds Settle", "family": "incomplete_pictures"},
            {"text": "Pencil Sketches on the Tape: Key Macro Indicators Still Rendering", "family": "pencil_sketches"},
            {"text": "Muted Data Transmission: Portfolios Rely on Conservative Baselines", "family": "muted_transmission"},
            {"text": "Signal Visibility Low: Monitors Await Fresh Pipeline Refreshes", "family": "visibility_low"},
            {"text": "Data Streams on Pause: Risk Monitors Keep Prudent Vigilance", "family": "streams_pause"},
            {"text": "Fog on the Horizon: Key Benchmarks Await Settlement Verification", "family": "fog_horizon"},
            {"text": "Signals in Transit: Asset Allocation Monitors Await Complete Prints", "family": "signals_transit"},
            {"text": "Sparse Signal Backdrop: Quantitative Dashboards Maintain Calm Stance", "family": "sparse_backdrop"},
        ],
    }

    @classmethod
    def detect_theme(
        cls,
        metrics: dict[str, CalculatedMetric],
        curve_regime: YieldCurveRegime | None,
        cross_asset_regime: RegimeClassification | None,
    ) -> str:
        has_metrics = bool(metrics and any(getattr(m, "quality_status", "") != "missing" for m in metrics.values()))
        has_valid_regime = bool(cross_asset_regime and cross_asset_regime.status == "OK")
        has_valid_curve = bool(curve_regime and curve_regime.status == "OK")

        if not has_metrics and not has_valid_regime and not has_valid_curve:
            return "INCONCLUSIVE_DATA"

        label = cross_asset_regime.label if has_valid_regime and cross_asset_regime else None
        if label == "Acute Stress":
            return "ACUTE_STRESS"
        if label == "Risk-Off":
            return "RISK_OFF"
        if label == "Strong Risk-On":
            return "STRONG_RISK_ON"
        if label == "Risk-On":
            return "RISK_ON"
        if label == "Mixed / Divergent":
            return "MIXED_DIVERGENT"

        if has_valid_curve and curve_regime:
            c_lbl = curve_regime.label
            if c_lbl in ("Bear Flattening", "Bear Steepening", "Parallel Upward Shift"):
                return "RATES_BEAR_MOVE"
            if c_lbl in ("Bull Flattening", "Bull Steepening", "Parallel Downward Shift"):
                return "RATES_BULL_MOVE"

        de10 = metrics.get("DE_10Y_YIELD")
        de2 = metrics.get("DE_2Y_YIELD")
        d10_val = abs(de10.change_value) if (de10 and de10.change_value is not None and getattr(de10, "quality_status", "") != "missing") else 0.0
        d2_val = abs(de2.change_value) if (de2 and de2.change_value is not None and getattr(de2, "quality_status", "") != "missing") else 0.0
        max_rate_chg = max(d10_val, d2_val)

        if max_rate_chg >= 4.0:
            net_dir = (de10.change_value if de10 and de10.change_value is not None else 0.0) + (
                de2.change_value if de2 and de2.change_value is not None else 0.0
            )
            return "RATES_BEAR_MOVE" if net_dir > 0 else "RATES_BULL_MOVE"

        if has_valid_regime or has_valid_curve or has_metrics:
            return "RANGE_BOUND"

        return "RANGE_BOUND"

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @classmethod
    def is_recent_duplicate(
        cls,
        candidate_text: str,
        candidate_family: str,
        recent_entries: list[tuple[date, str, str]],
        target_date: date,
    ) -> bool:
        norm_c = cls.normalize_text(candidate_text)
        words_c = set(re.findall(r"\w+", candidate_text.lower()))

        for hist_date, hist_text, hist_family in recent_entries:
            if not (0 < (target_date - hist_date).days <= 10):
                continue
            if norm_c == cls.normalize_text(hist_text):
                return True

            words_h = set(re.findall(r"\w+", hist_text.lower()))
            if words_c and words_h:
                jaccard = len(words_c & words_h) / len(words_c | words_h)
                if jaccard > 0.65:
                    return True

            if candidate_family and hist_family and candidate_family == hist_family and not candidate_family.startswith("dyn_"):
                return True

        return False

    @classmethod
    def load_history(cls, history_file: Path) -> dict[str, dict[str, str]]:
        if not history_file.exists():
            return {}
        try:
            data = json.loads(history_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                normalized: dict[str, dict[str, str]] = {}
                for k, v in data.items():
                    if isinstance(v, str):
                        normalized[k] = {"headline": v, "family": ""}
                    elif isinstance(v, dict) and "headline" in v:
                        normalized[k] = {"headline": str(v["headline"]), "family": str(v.get("family", ""))}
                return normalized
        except Exception as exc:
            logger.warning("Headline history corrupted or unreadable (%s). Starting clean.", exc)
        return {}

    @classmethod
    def save_history(cls, history_file: Path, history: dict[str, dict[str, str]]) -> None:
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = history_file.with_suffix(f".tmp_{os.getpid()}")
            tmp_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
            tmp_path.replace(history_file)
        except Exception as exc:
            logger.warning("Failed to atomically save headline history: %s", exc)

    @classmethod
    def _generate_dynamic_headline(
        cls,
        target_date: date,
        metrics: dict[str, CalculatedMetric],
        curve_regime: YieldCurveRegime | None,
        cross_asset_regime: RegimeClassification | None,
        recent_entries: list[tuple[date, str, str]],
    ) -> dict[str, str]:
        """
        Generates theme-aligned, punchy combinatoric headlines to guarantee uniqueness
        without dates, counters, or cross-theme fallback.
        """
        theme = cls.detect_theme(metrics, curve_regime, cross_asset_regime)
        
        theme_components: dict[str, tuple[list[str], list[str]]] = {
            "STRONG_RISK_ON": (
                ["Risk Appetite Surges", "Equities Lead the Charge", "Green Lights Across Desks", "Animal Spirits Unleashed"],
                ["While Credit Spreads Tighten", "As Volatility Drifts Lower", "With High Beta in High Gear", "As Allocators Expand Risk Budgets"],
            ),
            "RISK_ON": (
                ["Constructive Risk Tape", "Equities Push Ahead", "Supportive Market Currents", "Steady Bids on the Floor"],
                ["While Credit Stays Calm", "As Volatility Remains Muted", "With Carry Driving Portfolios", "As Spreads Behave Peacefully"],
            ),
            "ACUTE_STRESS": (
                ["Severe Risk Flight", "Defensive Helmets Fastened", "Turbulence Hits Trading Desks", "Liquidity Runs for Cover"],
                ["As Credit Spreads Blow Out", "While Volatility Surges Across Assets", "With Equity Bids in Retreat", "As Portfolios Brace for Drawdowns"],
            ),
            "RISK_OFF": (
                ["Defensive Rotation Underway", "Caution Guides the Morning", "Allocators Step on the Brakes", "Risk Assets Trim Sails"],
                ["While Havens Catch a Bid", "As Corporate Spreads Creep Wider", "With Equities Drifting Lower", "As Volatility Stirs Awake"],
            ),
            "RATES_BEAR_MOVE": (
                ["Duration Feels the Headwinds", "Benchmark Yields Step Higher", "Sovereign Selling Shapes the Tape", "Discount Rates Push Upward"],
                ["As Curve Bears Take Control", "While Bond Valuations Adjust", "With Term Premiums on the Rise", "As Fixed Income Reprices"],
            ),
            "RATES_BULL_MOVE": (
                ["Duration Catches Fresh Wind", "Benchmark Yields Ease Lower", "Sovereign Buyers Take the Lead", "Discount Rates Drift Down"],
                ["While Fixed Income Sleeves Rally", "As Core Curves Rebound", "With Bond Portfolios Catching Relief", "As Sovereign Pressure Subsides"],
            ),
            "MIXED_DIVERGENT": (
                ["Cross-Asset Signals Decouple", "Conflicting Currents on the Desk", "Asset Classes Part Ways", "Divergent Momentum Takes Hold"],
                ["As Equities and Credit Disagree", "While Markets Send Mixed Signals", "With Risk Sleeves Parting Ways", "As Cross-Asset Decoupling Deepens"],
            ),
            "RANGE_BOUND": (
                ["Range-Bound Drift Continues", "Subdued Tape Keeps Markets Quiet", "Asset Allocators Mark Time", "Equilibrium Governs the Morning"],
                ["While Macro Catalysts Stay Muted", "As Portfolios Hold Steady", "With Directional Moves on Pause", "As Trading Desks Await Fresh Sparks"],
            ),
            "INCONCLUSIVE_DATA": (
                ["Data Fog Over Trading Desks", "Signal Clarity Remains Muted", "Macro Feeds Keep Their Secrets", "Sparse Prints on the Morning Tape"],
                ["While Risk Monitors Stand Ready", "As Quantitative Engines Hold Ground", "With Allocation Models on Watch", "As Surveillance Systems Keep Vigilance"],
            ),
        }

        openers, connectors = theme_components.get(theme, theme_components["RANGE_BOUND"])
        combos = [f"{op}: {conn}" for op in openers for conn in connectors]

        rng = np.random.default_rng(target_date.toordinal())
        for idx in rng.permutation(len(combos)):
            text = combos[idx]
            fam = f"dyn_{theme.lower()}_{idx}"
            if not cls.is_recent_duplicate(text, fam, recent_entries, target_date):
                return {"text": text, "family": fam}

        fallback_text = combos[target_date.toordinal() % len(combos)]
        return {"text": fallback_text, "family": f"dyn_{theme.lower()}_fb"}

    @classmethod
    def generate(
        cls,
        target_date: date,
        metrics: dict[str, CalculatedMetric],
        curve_regime: YieldCurveRegime | None = None,
        cross_asset_regime: RegimeClassification | None = None,
        cache_dir: Path | str = ".cache",
    ) -> str:
        history_file = Path(cache_dir) / "headline_history.json"
        history = cls.load_history(history_file)
        target_str = target_date.isoformat()

        if target_str in history:
            cached_hl = history[target_str].get("headline", "")
            if cached_hl and cached_hl != "Morning Macro-Financial Risk Monitor" and "Morning Macro-Financial Risk Monitor" not in cached_hl:
                return cached_hl

        recent_entries: list[tuple[date, str, str]] = []
        for d_str, item in history.items():
            try:
                d = date.fromisoformat(d_str)
                if 0 < (target_date - d).days <= 10:
                    recent_entries.append((d, item.get("headline", ""), item.get("family", "")))
            except Exception:
                continue

        theme = cls.detect_theme(metrics, curve_regime, cross_asset_regime)
        candidates = list(cls.THEME_CANDIDATES.get(theme, cls.THEME_CANDIDATES["RANGE_BOUND"]))

        rng = np.random.default_rng(target_date.toordinal())
        chosen = None
        for idx in rng.permutation(len(candidates)):
            cand = candidates[idx]
            if not cls.is_recent_duplicate(cand["text"], cand["family"], recent_entries, target_date):
                chosen = cand
                break

        if not chosen:
            chosen = cls._generate_dynamic_headline(target_date, metrics, curve_regime, cross_asset_regime, recent_entries)

        history[target_str] = {"headline": chosen["text"], "family": chosen["family"]}
        cls.save_history(history_file, history)
        return chosen["text"]


class PortfolioInterpretationEngine:
    @staticmethod
    def generate_headline(
        target_date: date,
        metrics: dict[str, CalculatedMetric],
        curve_regime: YieldCurveRegime | None = None,
        cross_asset_regime: RegimeClassification | None = None,
        cache_dir: Path | str = ".cache",
    ) -> str:
        try:
            return HeadlineEngine.generate(target_date, metrics, curve_regime, cross_asset_regime, cache_dir)
        except Exception as exc:
            logger.error("Headline generation failed gracefully: %s", exc)
            return "Macro Radar Scanning: Markets Hum While Signals Recalibrate"

    @staticmethod
    def generate_narrative(metrics: dict[str, CalculatedMetric], curve_label: str) -> list[str]:
        bullets = []
        de10 = metrics.get("DE_10Y_YIELD")
        fi10 = metrics.get("FI_10Y_YIELD")
        fx = metrics.get("FX_EURUSD")
        hy = metrics.get("US_HY_OAS")
        vix = metrics.get("VIX_INDEX")

        # 1. EUR Sovereign Duration & Discount Rates
        if de10 and de10.latest_value is not None:
            chg_str = f"{de10.change_value:+.1f} bp" if de10.change_value is not None else "unchanged"
            bullets.append(
                f"Core EUR benchmark 10Y sovereign yield settled at {de10.latest_value:.2f}% ({chg_str}, curve: {curve_label}). "
                "Yield increases weigh on fixed income bond valuations while reducing regulatory pension liability present values."
            )

        # 2. Finland - Germany Sovereign Spread
        if fi10 and de10 and fi10.latest_value is not None and de10.latest_value is not None and fi10.quality_status != "missing" and de10.quality_status != "missing":
            spread = (fi10.latest_value - de10.latest_value) * 100.0
            date_diff = abs((fi10.as_of_date - de10.as_of_date).days) if (fi10.as_of_date and de10.as_of_date) else 0
            if date_diff <= 5:
                bullets.append(
                    f"Finland-Germany 10Y sovereign spread stands at {spread:+.1f} bp (FI: {fi10.latest_value:.2f}% as of {fi10.as_of_date}, DE: {de10.latest_value:.2f}% as of {de10.as_of_date}). "
                    "Reflects domestic Nordic sovereign credit premium over core benchmark."
                )

        # 3. Credit Spread Transmission
        if hy and hy.latest_value is not None:
            val_bp = hy.latest_value * 100.0 if hy.latest_value < 25.0 else hy.latest_value
            chg = f"{hy.change_value:+.1f} bp" if hy.change_value is not None else ""
            bullets.append(
                f"US High Yield OAS trades at {val_bp:.0f} bp ({chg}). "
                f"{'Spread widening indicates tighter corporate refinancing conditions.' if (hy.change_value or 0) > 4.0 else 'Credit spreads remain stable, supporting portfolio corporate credit returns.'}"
            )

        # 4. Currency Translation Channel
        if fx and fx.change_value is not None and abs(fx.change_value) >= 0.5:
            direction = "appreciation" if fx.change_value > 0 else "depreciation"
            bullets.append(
                f"EUR/USD 1D {direction} of {abs(fx.change_value):.2f}% alters the unhedged EUR valuation of USD-denominated assets."
            )

        # 5. Volatility Shock
        if vix and vix.latest_value is not None and vix.latest_value > 20.0:
            bullets.append(
                f"Equity market implied volatility elevated (VIX at {vix.latest_value:.1f}), signaling heightened macro risk aversion."
            )

        return bullets[:4]

    @staticmethod
    def generate_today_market_summary(
        metrics: dict[str, CalculatedMetric],
        curve_regime: YieldCurveRegime | None = None,
        cross_asset_regime: RegimeClassification | None = None,
        btc_surface: BTCVolatilitySurfaceResult | None = None,
    ) -> list[str]:
        """Generates 5 to 7 structured observations: observation, context, interpretation, risk, and limitation."""
        obs = []
        # 1. Sovereign Rates & Yield Curve Dynamics
        de10 = metrics.get("DE_10Y_YIELD")
        de2 = metrics.get("DE_2Y_YIELD")
        if de10 and de2 and de10.latest_value is not None and de2.latest_value is not None:
            c_lbl = curve_regime.label if curve_regime else "Range-bound"
            d10_str = f"{de10.change_value:+.1f} bp" if de10.change_value is not None else "0.0 bp"
            d2_str = f"{de2.change_value:+.1f} bp" if de2.change_value is not None else "0.0 bp"
            dur_sessions = curve_regime.duration_sessions if curve_regime else 1
            freq_1y = f"{curve_regime.frequency_1y:.1f}%" if (curve_regime and curve_regime.frequency_1y is not None) else "typical"
            obs.append(
                f"German 10Y sovereign yield settled at {de10.latest_value:.2f}% ({d10_str} 1D) while 2Y closed at {de2.latest_value:.2f}% ({d2_str} 1D). "
                f"This shift produces a {c_lbl} term structure dynamic that has persisted across {dur_sessions} session(s) and occurred on {freq_1y} of sessions over the past year. "
                f"Shifts in benchmark discount rates directly adjust regulatory pension liability present values while pressuring duration-sensitive bond valuations. "
                "Limitation: Yield curve shape reflects benchmark closing prints and does not isolate monetary policy expectations from term-premium shifts."
            )

        # 2. Corporate Credit OAS
        hy = metrics.get("US_HY_OAS")
        if hy and hy.latest_value is not None:
            val_bp = hy.latest_value * 100.0 if hy.latest_value < 25.0 else hy.latest_value
            hy_chg = f"{hy.change_value:+.1f} bp" if hy.change_value is not None else "0.0 bp"
            pctile_str = f"{hy.percentile_1y:.0f}th percentile" if hy.percentile_1y is not None else "median territory"
            obs.append(
                f"US High Yield OAS closed at {val_bp:.0f} bp ({hy_chg} 1D), placing the spread in the {pctile_str} of its one-year distribution. "
                f"Credit risk premia indicate {'tighter corporate refinancing conditions and risk aversion' if (hy.change_value or 0) > 2.0 else 'stable corporate balance-sheet liquidity and supportive credit carrying yield'}. "
                "Wider credit spreads erode corporate bond total returns while higher carrying yield supports reinvestment. "
                "Limitation: Option-adjusted spreads isolate credit risk relative to Treasuries but exclude single-issuer idiosyncratic default risk."
            )

        # 3. Equities Risk Appetite
        sp = metrics.get("EQ_SP500")
        if sp and sp.latest_value is not None:
            sp_chg = f"{sp.change_value:+.2f}%" if sp.change_value is not None else "-"
            pctile_eq = f"{sp.percentile_1y:.0f}th percentile" if sp.percentile_1y is not None else "its historical range"
            obs.append(
                f"US large-cap equity markets recorded {sp_chg} 1D (level: {sp.latest_value:.2f}), currently positioned in the {pctile_eq} of its 1Y range. "
                "Equity price action reflects prevailing institutional risk appetite and corporate earnings expectations across global asset allocators. "
                "Shifts in equity beta adjust total portfolio return buffers and growth-sleeve asset values. "
                "Limitation: Realized daily price changes do not incorporate forward valuation multiples or equity risk premium expansions."
            )

        # 4. Foreign Exchange (EUR/USD)
        fx = metrics.get("FX_EURUSD")
        if fx and fx.latest_value is not None:
            dir_str = "euro appreciation" if (fx.change_value or 0) > 0 else "euro depreciation"
            pct_str = f"{fx.percentile_1y:.0f}th percentile" if fx.percentile_1y is not None else "its historical range"
            chg_str = f"{abs(fx.change_value):.2f}%" if fx.change_value is not None else "-"
            obs.append(
                f"EUR/USD reference rate trades at {fx.latest_value:.4f} USD per EUR ({chg_str} 1D, reflecting {dir_str}) in the {pct_str} of its 1-year historical range. "
                "Fluctuations in the foreign exchange rate adjust the unhedged EUR valuation of USD-denominated sovereign and corporate assets. "
                "Currency translation moves alter balance-sheet solvency ratios for institutions holding unhedged international exposures. "
                "Limitation: Spot reference rates reflect settlement pricing and do not imply purchasing power parity or currency fair value."
            )

        # 5. Implied Volatility (VIX)
        vix = metrics.get("VIX_INDEX")
        if vix and vix.latest_value is not None:
            v_chg = f"{vix.change_value:+.1f}%" if vix.change_value is not None else "-"
            zone = "subdued" if vix.latest_value < 15 else ("moderate" if vix.latest_value < 20 else "elevated")
            obs.append(
                f"CBOE Volatility Index (VIX) closed at {vix.latest_value:.2f} ({v_chg} 1D), holding within the {zone} volatility context zone. "
                "Options-market pricing reflects near-term S&P 500 dispersion demand and tail-risk protection appetite among institutional hedgers. "
                "Elevated volatility bands increase option-hedging premium costs and can trigger risk-parity de-leveraging. "
                "Limitation: The VIX reflects 30-day option-implied expected variance and carries no directional equity price forecast."
            )

        return obs[:6]

    @staticmethod
    def interpret_eurusd(metric: CalculatedMetric | None) -> str:
        """Dynamic interpretation for the standalone EUR/USD chart."""
        if not metric or metric.latest_value is None:
            return "EUR/USD historical exchange rate data is currently unavailable from the European Central Bank."
        
        chg_1d = metric.change_value
        chg_5d = metric.pct_5d
        pctile = metric.percentile_1y
        z = metric.zscore_3y

        sentences = [
            f"EUR/USD closed at {metric.latest_value:.4f} USD per EUR as of {metric.as_of_date}."
        ]
        if chg_1d is not None and chg_5d is not None:
            action_1d = "euro appreciation" if chg_1d >= 0 else "euro depreciation"
            sentences.append(
                f"The exchange rate moved {chg_1d:+.2f}% over the latest session (reflecting {action_1d}) and {chg_5d:+.2f}% over five sessions."
            )
        if pctile is not None and z is not None:
            pos_wording = "the lower quartile of" if pctile <= 25 else ("the upper quartile of" if pctile >= 75 else "the middle of")
            sentences.append(f"The spot rate sits in {pos_wording} its 1-year historical range ({pctile:.0f}th percentile, 3Y z-score of {z:+.1f}σ).")
        sentences.append(
            "Higher values denote euro appreciation against the dollar, which reduces the EUR valuation of unhedged USD assets, while historical distribution rank alone does not imply currency fair value."
        )
        return " ".join(sentences)

    @staticmethod
    def interpret_vix(metric: CalculatedMetric | None) -> str:
        """Dynamic interpretation for the standalone VIX chart."""
        if not metric or metric.latest_value is None:
            return "CBOE Volatility Index data is currently unavailable."
        
        sentences = [
            f"The CBOE VIX settled at {metric.latest_value:.2f} as of {metric.as_of_date}."
            f"{' VIX above 30 signals acute market stress.' if metric.latest_value >= 30 else (' VIX below 15 reflects subdued implied volatility.' if metric.latest_value <= 15 else '')}"
        ]
        if metric.change_value is not None:
            trend_desc = "reflecting increased option-implied uncertainty" if metric.change_value > 0 else "reflecting reduced option-implied uncertainty"
            sentences.append(f"The index shifted {metric.change_value:+.1f}% over the latest session, {trend_desc}.")
        if metric.percentile_1y is not None:
            sentences.append(
                f"Implied volatility resides in the {metric.percentile_1y:.0f}th percentile of its 1-year historical distribution."
            )
        sentences.append(
            "Configured heuristic context bands: subdued (<15), moderate (15–20), elevated (20–30), and high stress (>30)."
        )
        sentences.append(
            "The VIX is calculated from S&P 500 index option prices and reflects expected volatility; it does not forecast directional equity price trends."
        )
        return " ".join(sentences)

    @staticmethod
    def interpret_sovereign_curves(curve_regime: YieldCurveRegime | None) -> str:
        if not curve_regime:
            return "Sovereign term structure observations are pending validation."
        return (
            f"Current sovereign dynamics display {curve_regime.label}. {curve_regime.plain_language_definition} "
            f"Key risk implication: {curve_regime.risk_implication}"
        )

    @staticmethod
    def interpret_credit(hy_metric: CalculatedMetric | None) -> str:
        if not hy_metric or hy_metric.latest_value is None:
            return "US corporate OAS data currently unavailable."
        val = hy_metric.latest_value
        val_bp = val * 100.0 if val < 25.0 else val
        chg = f"{hy_metric.change_value:+.1f} bp" if hy_metric.change_value is not None else "unchanged"
        pctile_str = f" in the {hy_metric.percentile_1y:.0f}th percentile of its 1-year distribution" if hy_metric.percentile_1y is not None else ""
        trend = "widened, indicating tighter refinancing conditions" if (hy_metric.change_value or 0) > 2.0 else "contracted or held steady, supporting credit carrying yield"
        return (
            f"US High Yield OAS sits at {val_bp:.0f} bp ({chg} 1D){pctile_str}. Spreads have {trend}. "
            "Option-adjusted spreads isolate pure corporate credit risk premium above duration-matched Treasuries."
        )

    @staticmethod
    def interpret_btc_surface(surface: BTCVolatilitySurfaceResult | None) -> str:
        if not surface or surface.valid_observation_count == 0:
            return "Deribit BTC option market volatility surface is currently unavailable."
        tau_min = min(surface.retained_expiry_maturities) if surface.retained_expiry_maturities else 0.02
        tau_max = max(surface.retained_expiry_maturities) if surface.retained_expiry_maturities else 1.0
        rmse_str = f"Weighted calibration RMSE is {surface.weighted_rmse:.2f}%." if surface.weighted_rmse else ""
        return (
            f"The BTC/USD surface is calibrated directly from {surface.retained_quote_count} active Deribit option quotes spanning "
            f"{len(surface.retained_expiries)} expiries ({tau_min*365:.0f}d to {tau_max*365:.0f}d). "
            f"Quasi-explicit SVI parameterizations model the strike smile, while monotone PCHIP splines enforce term structure consistency. "
            f"{rmse_str} The surface represents market-implied volatility expectations and carries no directional price forecast."
        )