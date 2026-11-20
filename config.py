"""Application configuration loader."""

from pathlib import Path
import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator
from morning_report.models import SeriesDefinition


class DeribitVolatilitySurfaceConfig(BaseModel):
    enabled: bool = True
    currency: str = "BTC"
    kind: str = "option"
    minimum_days_to_expiry: float = 7.0
    maximum_days_to_expiry: float = 365.0
    minimum_expiries: int = 4
    minimum_quotes_per_expiry: int = 7
    preferred_quotes_per_expiry: int = 15
    minimum_total_quotes: int = 80
    minimum_open_interest: float = 0.0
    minimum_volume: float = 0.0
    forward_moneyness_min: float = 0.60
    forward_moneyness_max: float = 1.40
    maximum_relative_bid_ask_spread: float = 0.50
    maximum_mark_mid_deviation: float = 0.30
    minimum_iv_pct: float = 5.0
    maximum_iv_pct: float = 300.0
    maximum_data_age_minutes: float = 30.0
    use_mark_iv: bool = True
    require_two_sided_market: bool = False
    cache_ttl_minutes: int = 10
    multistart_count: int = 12
    dense_maturity_points: int = 60
    dense_moneyness_points: int = 81
    call_put_discrepancy_threshold_pts: float = 3.0

    @model_validator(mode="after")
    def validate_config(self) -> "DeribitVolatilitySurfaceConfig":
        if self.minimum_days_to_expiry >= self.maximum_days_to_expiry:
            raise ValueError("minimum_days_to_expiry must be < maximum_days_to_expiry")
        if self.forward_moneyness_min >= self.forward_moneyness_max:
            raise ValueError("forward_moneyness_min must be < forward_moneyness_max")
        if self.dense_maturity_points < 20 or self.dense_moneyness_points < 21:
            raise ValueError("Dense grid must have >= 20 maturity points and >= 21 moneyness points.")
        return self


class AppSettings(BaseModel):
    app_name: str = "Morning Macro-Financial Risk Monitor"
    timezone: str = "Europe/Helsinki"
    base_currency: str = "EUR"
    report_time: str = "07:30"
    output_directory: str = "output"
    minimum_history_for_zscore: int = 252
    default_zscore_window: int = 756
    default_volatility_window: int = 60
    max_news_items: int = 5
    http_timeout_seconds: int = 15
    cache_dir: str = Field(default=".cache", validation_alias=AliasChoices("cache_dir", "cache_directory"))
    deribit_volatility_surface: DeribitVolatilitySurfaceConfig = Field(default_factory=DeribitVolatilitySurfaceConfig)


class ThresholdLevel(BaseModel):
    amber: float
    red: float


class PlausibilityThresholds(BaseModel):
    max_equity_daily_move_pct: float = 25.0
    max_rates_daily_move_bp: float = 100.0
    max_credit_daily_move_bp: float = 300.0
    max_fx_daily_move_pct: float = 10.0


class ThresholdConfig(BaseModel):
    daily_zscore: ThresholdLevel
    rates_move_bp: ThresholdLevel
    sovereign_spread_move_bp: ThresholdLevel
    credit_spread_move_bp: ThresholdLevel
    equity_return_pct: ThresholdLevel
    fx_return_pct: ThresholdLevel
    commodities_return_pct: ThresholdLevel | None = None
    volatility_change_pct: ThresholdLevel | None = None
    plausibility: PlausibilityThresholds


class ConfigBundle:
    def __init__(self, root_dir: Path | None = None) -> None:
        if root_dir is not None:
            self.root_dir = root_dir
        else:
            current = Path(__file__).resolve().parent
            if (current / "config").exists() or (current / "settings.yaml").exists():
                self.root_dir = current
            elif (current.parent.parent / "config").exists():
                self.root_dir = current.parent.parent
            else:
                self.root_dir = current
        cfg_path = self.root_dir / "config" if (self.root_dir / "config").exists() else self.root_dir
        
        settings_file = cfg_path / "settings.yaml"
        self.settings = AppSettings(**(yaml.safe_load(settings_file.read_text(encoding="utf-8")) or {})) if settings_file.exists() else AppSettings()
        self.deribit_volatility_surface = self.settings.deribit_volatility_surface
        
        thresholds_file = cfg_path / "thresholds.yaml"
        if thresholds_file.exists():
            self.thresholds = ThresholdConfig(**(yaml.safe_load(thresholds_file.read_text(encoding="utf-8")) or {}))
        else:
            self.thresholds = ThresholdConfig(
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
            
        series_file = cfg_path / "series.yaml"
        if series_file.exists():
            series_data = yaml.safe_load(series_file.read_text(encoding="utf-8")) or {}
            self.series = [SeriesDefinition(**item) for item in series_data.get("series", [])]
        else:
            self.series = self._build_canonical_series()
            
        news_file = cfg_path / "news_sources.yaml"
        self.news_sources = yaml.safe_load(news_file.read_text(encoding="utf-8")) if news_file.exists() else {
            "feeds": [
                {"name": "Federal Reserve", "url": "https://www.federalreserve.gov/feeds/press_all.xml", "category": "Central Bank", "authority_weight": 1.0},
                {"name": "European Central Bank", "url": "https://www.ecb.europa.eu/rss/press.html", "category": "Central Bank", "authority_weight": 1.0},
            ],
            "relevance_keywords": ["rates", "monetary", "inflation", "yield", "bonds", "financial stability", "credit"]
        }

    @staticmethod
    def _build_canonical_series() -> list[SeriesDefinition]:
        """Construct the canonical institutional multi-asset and sovereign term structure series."""
        series: list[SeriesDefinition] = []
        # German Bund curve (ECB SDMX)
        de_tenors = [("6M", 0.5), ("1Y", 1.0), ("2Y", 2.0), ("3Y", 3.0), ("5Y", 5.0), ("7Y", 7.0), ("10Y", 10.0), ("15Y", 15.0), ("20Y", 20.0), ("30Y", 30.0)]
        for tenor_label, years in de_tenors:
            series.append(SeriesDefinition(
                id=f"DE_{tenor_label}_YIELD", display_name=f"German Bund {tenor_label} Yield", asset_class="rates",
                provider="ecb", provider_series_id=f"YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{tenor_label}",
                unit="pct", transformation="diff_bp", country="DE", tenor=tenor_label, tenor_years=years, curve_group="DE_BUND"
            ))
        # Finland sovereign (ECB SDMX)
        series.append(SeriesDefinition(id="FI_10Y_YIELD", display_name="Finland 10Y Benchmark Yield", asset_class="rates", provider="ecb", provider_series_id="FM.D.FI.EUR.4F.BB.U2.10Y.SR.A", unit="pct", transformation="diff_bp", country="FI", tenor="10Y", tenor_years=10.0, curve_group="FI_GOVT"))
        series.append(SeriesDefinition(id="FI_5Y_YIELD", display_name="Finland 5Y Benchmark Yield", asset_class="rates", provider="ecb", provider_series_id="FM.D.FI.EUR.4F.BB.U2.5Y.SR.A", unit="pct", transformation="diff_bp", country="FI", tenor="5Y", tenor_years=5.0, curve_group="FI_GOVT"))
        series.append(SeriesDefinition(id="FI_2Y_YIELD", display_name="Finland 2Y Benchmark Yield", asset_class="rates", provider="ecb", provider_series_id="FM.D.FI.EUR.4F.BB.U2.2Y.SR.A", unit="pct", transformation="diff_bp", country="FI", tenor="2Y", tenor_years=2.0, curve_group="FI_GOVT"))
        # US Treasury Curve (FRED)
        us_tenors = [("6M", "DGS6MO", 0.5), ("1Y", "DGS1", 1.0), ("2Y", "DGS2", 2.0), ("3Y", "DGS3", 3.0), ("5Y", "DGS5", 5.0), ("7Y", "DGS7", 7.0), ("10Y", "DGS10", 10.0), ("20Y", "DGS20", 20.0), ("30Y", "DGS30", 30.0)]
        for tenor_label, fred_id, years in us_tenors:
            series.append(SeriesDefinition(
                id=f"US_{tenor_label}_YIELD", display_name=f"US Treasury {tenor_label} Yield", asset_class="rates",
                provider="fred", provider_series_id=fred_id, unit="pct", currency="USD", transformation="diff_bp", country="US", tenor=tenor_label, tenor_years=years, curve_group="US_TREASURY"
            ))
        # Credit OAS & Indices
        series.append(SeriesDefinition(id="US_HY_OAS", display_name="US Corporate HY OAS", asset_class="credit", provider="fred", provider_series_id="BAMLH0A0HYM2", unit="bp", currency="USD", transformation="diff_bp", is_in_basis_points=False))
        series.append(SeriesDefinition(id="US_IG_OAS", display_name="US Corporate IG OAS", asset_class="credit", provider="fred", provider_series_id="BAMLC0A0CM", unit="bp", currency="USD", transformation="diff_bp", is_in_basis_points=False))
        series.append(SeriesDefinition(id="EUR_HY_OAS", display_name="Euro High Yield OAS", asset_class="credit", provider="fred", provider_series_id="BAMLHE00EHYIOAS", unit="bp", currency="EUR", transformation="diff_bp", is_in_basis_points=False, is_proxy=True, proxy_description="ICE BofA Euro High Yield OAS"))
        # FX, Volatility, Equities
        series.append(SeriesDefinition(id="FX_EURUSD", display_name="EUR / USD Reference Rate", asset_class="fx", provider="ecb", provider_series_id="EXR.D.USD.EUR.SP00.A", unit="USD/EUR", transformation="pct_change", secondary_provider="yfinance"))
        series.append(SeriesDefinition(id="VIX_INDEX", display_name="CBOE Volatility Index (VIX)", asset_class="volatility", provider="fred", provider_series_id="VIXCLS", unit="Index", transformation="diff_level", secondary_provider="yfinance"))
        series.append(SeriesDefinition(id="EQ_EUROPE_600", display_name="STOXX Europe 600", asset_class="equities", provider="ecb", provider_series_id="FM.D.U2.EUR.4F.KR.ESTX50.A", unit="Index", transformation="pct_change", is_proxy=True, proxy_description="Euro Stoxx 50 ECB benchmark proxy"))
        return series