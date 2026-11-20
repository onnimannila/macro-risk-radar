"""Data acquisition adapters for ECB SDMX, FRED, Deribit Options, US Treasury, Market Data, Caching, and RSS."""

from datetime import date, datetime, timezone, timedelta
import html
import io
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
import xml.etree.ElementTree as ET
import feedparser
import httpx
import numpy as np
import pandas as pd
from morning_report.config import DeribitVolatilitySurfaceConfig
from morning_report.models import NewsItem, OptionMarketObservation, SeriesDefinition

logger = logging.getLogger(__name__)


class DiskCacheManager:
    """Thread-safe disk caching for raw provider responses and normalized series."""
    def __init__(self, cache_dir: Path | str = ".cache", ttl_hours: int = 18) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def get(self, key: str) -> pd.DataFrame | None:
        file_path = self.cache_dir / f"{key}.csv"
        meta_path = self.cache_dir / f"{key}.meta.json"
        if not file_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(meta.get("cached_at", "1970-01-01T00:00:00"))
            if datetime.now(timezone.utc) - cached_at > self.ttl:
                return None
            df = pd.read_csv(file_path)
            if "observation_date" in df.columns:
                df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
            return df
        except Exception as exc:
            logger.warning("Cache read failed for key %s: %s", key, exc)
            return None

    def set(self, key: str, df: pd.DataFrame, provider: str) -> None:
        if df.empty:
            return
        try:
            file_path = self.cache_dir / f"{key}.csv"
            meta_path = self.cache_dir / f"{key}.meta.json"
            df.to_csv(file_path, index=False)
            meta = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "row_count": len(df),
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception as exc:
            logger.warning("Cache write failed for key %s: %s", key, exc)


class ECBDataProvider:
    BASE_URL = "https://data-api.ecb.europa.eu/service/data"

    def __init__(self, timeout: int = 15, max_retries: int = 3, cache: DiskCacheManager | None = None) -> None:
        self.client = httpx.Client(timeout=timeout)
        self.max_retries = max_retries
        self.cache = cache or DiskCacheManager()
        self.last_status: dict[str, str] = {}
        self.last_error: dict[str, str] = {}

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "ECBDataProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def fetch_series(self, definition: SeriesDefinition, start_date: date, end_date: date) -> pd.DataFrame:
        cache_key = f"ecb_{definition.id}_{start_date.isoformat()}_{end_date.isoformat()}"
        cached_df = self.cache.get(cache_key)
        if cached_df is not None:
            self.last_status[definition.id] = "CACHE_HIT"
            return cached_df

        parts = definition.provider_series_id.split(".", 1)
        if len(parts) != 2:
            self.last_status[definition.id] = "INVALID_ID"
            self.last_error[definition.id] = f"Malformed ECB series key '{definition.provider_series_id}'"
            return pd.DataFrame()
        flow, key = parts
        for attempt in range(1, self.max_retries + 1):
            try:
                # Attempt fast, standard CSV data format first
                csv_resp = self.client.get(
                    f"{self.BASE_URL}/{flow}/{key}",
                    params={"startPeriod": start_date.isoformat(), "endPeriod": end_date.isoformat(), "format": "csvdata"},
                )
                if csv_resp.status_code == 200 and "TIME_PERIOD" in csv_resp.text:
                    df = self._parse_csv(csv_resp.text, definition)
                    if not df.empty:
                        self.cache.set(cache_key, df, "ecb_csv")
                        self.last_status[definition.id] = "ONLINE_SUCCESS"
                        return df
                elif csv_resp.status_code in [400, 404]:
                    self.last_status[definition.id] = f"HTTP_{csv_resp.status_code}"

                # Fallback to SDMX generic XML
                resp = self.client.get(
                    f"{self.BASE_URL}/{flow}/{key}",
                    params={"startPeriod": start_date.isoformat(), "endPeriod": end_date.isoformat(), "format": "genericdata"},
                    headers={"Accept": "application/vnd.sdmx.genericdata+xml;version=2.1"},
                )
                if resp.status_code == 200:
                    df = self._parse_xml(resp.text, definition)
                    if not df.empty:
                        self.cache.set(cache_key, df, "ecb")
                        self.last_status[definition.id] = "ONLINE_SUCCESS"
                        return df
                    self.last_status[definition.id] = "EMPTY_PAYLOAD"
                    self.last_error[definition.id] = "No observation tags matched in SDMX XML response."
                    return pd.DataFrame()
                elif resp.status_code in [404, 400]:
                    self.last_status[definition.id] = f"HTTP_{resp.status_code}"
                    self.last_error[definition.id] = f"ECB API returned status {resp.status_code}"
                    return pd.DataFrame()
                time.sleep(0.3 * attempt)
            except httpx.RequestError as exc:
                self.last_status[definition.id] = "NETWORK_ERROR"
                self.last_error[definition.id] = str(exc)
                time.sleep(0.3 * attempt)
            except Exception as exc:
                self.last_status[definition.id] = "PARSE_ERROR"
                self.last_error[definition.id] = str(exc)
                return pd.DataFrame()
        return pd.DataFrame()

    def _parse_csv(self, csv_text: str, definition: SeriesDefinition) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.StringIO(csv_text))
            if "TIME_PERIOD" in df.columns and "OBS_VALUE" in df.columns:
                clean = df.dropna(subset=["TIME_PERIOD", "OBS_VALUE"]).copy()
                clean = clean[clean["OBS_VALUE"].astype(str).str.strip() != "."]
                clean["observation_date"] = pd.to_datetime(clean["TIME_PERIOD"]).dt.date
                clean["value"] = pd.to_numeric(clean["OBS_VALUE"], errors="coerce")
                clean = clean.dropna(subset=["value"])
                clean["unit"] = definition.unit
                clean["currency"] = definition.currency
                clean = clean.sort_values("observation_date").drop_duplicates(subset=["observation_date"])
                return clean[["observation_date", "value", "unit", "currency"]]
        except Exception as exc:
            logger.debug("ECB CSV parse failed: %s", exc)
        return pd.DataFrame()

    def _parse_xml(self, xml_text: str, definition: SeriesDefinition) -> pd.DataFrame:
        ns = {"g": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic"}
        root = ET.fromstring(xml_text)
        rows = []
        for obs in root.findall(".//g:Obs", ns):
            d = obs.find("./g:ObsDimension", ns)
            v = obs.find("./g:ObsValue", ns)
            if d is not None and v is not None:
                try:
                    rows.append({
                        "observation_date": pd.to_datetime(d.attrib.get("value")).date(),
                        "value": float(v.attrib.get("value")),
                        "unit": definition.unit,
                        "currency": definition.currency,
                    })
                except Exception:
                    continue
        return pd.DataFrame(rows)

    def health_check(self) -> bool:
        try:
            return self.client.get(f"{self.BASE_URL}/EXR/D.USD.EUR.SP00.A?lastNObservations=1", timeout=5).status_code in [200, 304]
        except Exception:
            return False


class FREDDataProvider:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    CSV_FALLBACK_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, timeout: int = 15, max_retries: int = 3, cache: DiskCacheManager | None = None) -> None:
        self.api_key = os.getenv("FRED_API_KEY", "")
        self.client = httpx.Client(timeout=timeout)
        self.max_retries = max_retries
        self.cache = cache or DiskCacheManager()
        self.last_status: dict[str, str] = {}
        self.last_error: dict[str, str] = {}
        self.cache_status: dict[str, str] = {}

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "FREDDataProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def fetch_series(self, definition: SeriesDefinition, start_date: date, end_date: date) -> pd.DataFrame:
        cache_key = f"fred_{definition.id}_{start_date.isoformat()}_{end_date.isoformat()}"
        cached_df = self.cache.get(cache_key)
        if cached_df is not None:
            self.last_status[definition.id] = "CACHE_HIT"
            self.cache_status[definition.id] = "HIT"
            return cached_df

        self.cache_status[definition.id] = "MISS"
        # Route 1: Official API Key (when configured)
        if self.api_key:
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = self.client.get(self.BASE_URL, params={
                        "series_id": definition.provider_series_id,
                        "api_key": self.api_key,
                        "file_type": "json",
                        "observation_start": start_date.isoformat(),
                        "observation_end": end_date.isoformat(),
                    })
                    if resp.status_code == 200:
                        data = resp.json()
                        df = pd.DataFrame([
                            {"observation_date": pd.to_datetime(i["date"]).date(), "value": float(i["value"]), "unit": definition.unit, "currency": definition.currency}
                            for i in data.get("observations", []) if i.get("value") and i.get("value") != "."
                        ])
                        if not df.empty:
                            self.cache.set(cache_key, df, "fred")
                            self.last_status[definition.id] = "ONLINE_SUCCESS"
                            return df
                    elif resp.status_code in [400, 404]:
                        self.last_status[definition.id] = f"HTTP_{resp.status_code}"
                        break
                    time.sleep(0.3 * attempt)
                except Exception as exc:
                    time.sleep(0.3 * attempt)

        # Route 2: Free St. Louis Fed direct CSV download
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.get(self.CSV_FALLBACK_URL, params={"id": definition.provider_series_id})
                if resp.status_code == 200 and resp.text:
                    df = pd.read_csv(io.StringIO(resp.text))
                    if len(df.columns) >= 2:
                        d_col, v_col = df.columns[0], df.columns[1]
                        df = df[df[v_col] != "."].dropna()
                        df["observation_date"] = pd.to_datetime(df[d_col]).dt.date
                        df["value"] = pd.to_numeric(df[v_col], errors="coerce")
                        df = df.dropna(subset=["value"])
                        df = df[(df["observation_date"] >= start_date) & (df["observation_date"] <= end_date)]
                        df["unit"] = definition.unit
                        df["currency"] = definition.currency
                        res = df[["observation_date", "value", "unit", "currency"]]
                        if not res.empty:
                            self.cache.set(cache_key, res, "fred_csv")
                            self.last_status[definition.id] = "ONLINE_SUCCESS"
                            return res
                time.sleep(0.3 * attempt)
            except Exception as exc:
                self.last_error[definition.id] = str(exc)
                time.sleep(0.3 * attempt)

        if not self.api_key:
            self.last_status[definition.id] = "MISSING_API_KEY"
            self.last_error[definition.id] = "FRED_API_KEY is not configured and direct keyless download failed."
        return pd.DataFrame()

    def health_check(self) -> bool:
        return bool(self.api_key)


class YahooFinanceDataProvider:
    """Secondary market data provider using Yahoo Finance for tradable instruments."""
    def __init__(self, cache: DiskCacheManager | None = None) -> None:
        self.cache = cache or DiskCacheManager()
        self.last_status: dict[str, str] = {}
        self.last_error: dict[str, str] = {}
        self.cache_status: dict[str, str] = {}

    def fetch_series(self, definition: SeriesDefinition, start_date: date, end_date: date) -> pd.DataFrame:
        ticker_map = {"FX_EURUSD": "EURUSD=X", "VIX_INDEX": "^VIX", "EQ_SP500": "^GSPC", "EQ_EUROPE_600": "^STOXX"}
        ticker_symbol = ticker_map.get(definition.id, definition.provider_series_id)
        cache_key = f"yf_{definition.id}_{start_date.isoformat()}_{end_date.isoformat()}"
        cached_df = self.cache.get(cache_key)
        if cached_df is not None:
            self.last_status[definition.id] = "CACHE_HIT"
            self.cache_status[definition.id] = "HIT"
            return cached_df

        self.cache_status[definition.id] = "MISS"
        try:
            import yfinance as yf
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat())
            if not hist.empty:
                hist = hist.reset_index()
                date_col = "Date" if "Date" in hist.columns else hist.columns[0]
                hist["observation_date"] = pd.to_datetime(hist[date_col]).dt.date
                hist["value"] = pd.to_numeric(hist["Close"], errors="coerce")
                hist = hist.dropna(subset=["value"])
                hist["unit"] = definition.unit
                hist["currency"] = definition.currency
                res = hist[["observation_date", "value", "unit", "currency"]]
                self.cache.set(cache_key, res, "yfinance")
                self.last_status[definition.id] = "ONLINE_SUCCESS"
                return res
            self.last_status[definition.id] = "EMPTY_PAYLOAD"
            self.last_error[definition.id] = "Yahoo Finance returned no history"
        except Exception as exc:
            self.last_status[definition.id] = "PROVIDER_ERROR"
            self.last_error[definition.id] = str(exc)
        return pd.DataFrame()


class DeribitOptionsProvider:
    """
    Direct unauthenticated market data provider for Deribit BTC options.
    Queries official production API v2 for active instruments and book summaries.
    """
    BASE_URL = "https://www.deribit.com/api/v2"

    def __init__(self, config: DeribitVolatilitySurfaceConfig | None = None, cache_dir: Path | str = ".cache") -> None:
        self.config = config or DeribitVolatilitySurfaceConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": "MorningMacroRiskMonitor/1.0 (Institutional Risk Team)"},
        )
        self.last_status: str = "PENDING"
        self.last_error: str | None = None
        self.last_retrieval_utc: datetime | None = None
        self.cache_status: str = "MISS"

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "DeribitOptionsProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def health_check(self) -> bool:
        try:
            resp = self.client.get(f"{self.BASE_URL}/public/test", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def fetch_btc_option_chain(self, execution_time: datetime | None = None) -> tuple[list[OptionMarketObservation], dict[str, Any]]:
        """
        Fetches and joins active Deribit BTC option instruments with book summary quotes.
        Returns (valid_and_excluded_observations, metadata_diagnostics).
        """
        now_utc = execution_time or datetime.now(timezone.utc)
        cache_file = self.cache_dir / f"deribit_btc_chain_{now_utc.strftime('%Y%m%d_%H%M')}.json"

        # Disk cache lookup within configured cache_ttl_minutes
        cached_payload = self._read_recent_cache(now_utc)
        if cached_payload is not None:
            self.cache_status = "HIT"
            self.last_status = "ONLINE_SUCCESS"
            return self._parse_chain_payload(cached_payload, now_utc)

        self.cache_status = "MISS"
        # Step 1: Discover active BTC option instruments
        instruments_data = []
        for attempt in range(1, 4):
            try:
                resp = self.client.get(
                    f"{self.BASE_URL}/public/get_instruments",
                    params={"currency": self.config.currency, "kind": self.config.kind, "expired": "false"},
                )
                if resp.status_code == 200:
                    instruments_data = resp.json().get("result", [])
                    break
                time.sleep(0.3 * attempt)
            except Exception as exc:
                self.last_error = f"Instruments call failed: {exc}"
                time.sleep(0.4 * attempt)

        if not instruments_data:
            self.last_status = "HTTP_ERROR"
            return [], {"error": self.last_error or "Empty instruments payload from Deribit"}

        # Step 2: Retrieve bulk book summaries
        summaries_data = []
        for attempt in range(1, 4):
            try:
                resp = self.client.get(
                    f"{self.BASE_URL}/public/get_book_summary_by_currency",
                    params={"currency": self.config.currency, "kind": self.config.kind},
                )
                if resp.status_code == 200:
                    summaries_data = resp.json().get("result", [])
                    break
                time.sleep(0.3 * attempt)
            except Exception as exc:
                self.last_error = f"Book summary call failed: {exc}"
                time.sleep(0.4 * attempt)

        if not summaries_data:
            self.last_status = "HTTP_ERROR"
            return [], {"error": self.last_error or "Empty book summary payload from Deribit"}

        payload = {
            "retrieval_timestamp": now_utc.isoformat(),
            "instruments": instruments_data,
            "summaries": summaries_data,
        }
        try:
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write Deribit cache: %s", exc)

        self.last_status = "ONLINE_SUCCESS"
        self.last_retrieval_utc = now_utc
        return self._parse_chain_payload(payload, now_utc)

    def _read_recent_cache(self, now_utc: datetime) -> dict[str, Any] | None:
        try:
            candidates = sorted(self.cache_dir.glob("deribit_btc_chain_*.json"), reverse=True)
            if not candidates:
                return None
            latest = candidates[0]
            text = latest.read_text(encoding="utf-8")
            data = json.loads(text)
            cached_dt = datetime.fromisoformat(data["retrieval_timestamp"])
            age_minutes = (now_utc - cached_dt).total_seconds() / 60.0
            if age_minutes <= self.config.cache_ttl_minutes:
                return data
        except Exception as exc:
            logger.debug("Deribit cache read check failed: %s", exc)
        return None

    def _parse_chain_payload(self, payload: dict[str, Any], now_utc: datetime) -> tuple[list[OptionMarketObservation], dict[str, Any]]:
        summary_map = {item["instrument_name"]: item for item in payload.get("summaries", [])}
        observations: list[OptionMarketObservation] = []
        retrieval_dt = datetime.fromisoformat(payload.get("retrieval_timestamp", now_utc.isoformat()))

        spot_est = 0.0
        for inst in payload.get("instruments", []):
            name = inst.get("instrument_name", "")
            sum_data = summary_map.get(name, {})
            strike = float(inst.get("strike", 0.0))
            opt_type = "call" if inst.get("option_type") == "call" or name.endswith("-C") else "put"
            exp_ts_ms = inst.get("expiration_timestamp")
            if not exp_ts_ms:
                continue
            expiry_dt = datetime.fromtimestamp(exp_ts_ms / 1000.0, tz=timezone.utc)
            time_to_exp_sec = (expiry_dt - retrieval_dt).total_seconds()
            tau = time_to_exp_sec / (365.25 * 86400.0)
            days_to_exp = time_to_exp_sec / 86400.0

            # Deribit book summary provides underlying_price or estimated_delivery_price (Forward reference)
            fwd = float(sum_data.get("estimated_delivery_price") or sum_data.get("underlying_price") or 0.0)
            und_price = float(sum_data.get("underlying_price") or fwd)
            if und_price > spot_est:
                spot_est = und_price

            # Mark IV: Deribit returns mark_iv in percentage points (e.g. 58.4 -> 0.584)
            raw_mark_iv = sum_data.get("mark_iv")
            mark_iv = (float(raw_mark_iv) / 100.0) if raw_mark_iv is not None and float(raw_mark_iv) > 0 else None
            raw_bid_iv = sum_data.get("bid_iv")
            bid_iv = (float(raw_bid_iv) / 100.0) if raw_bid_iv is not None and float(raw_bid_iv) > 0 else None
            raw_ask_iv = sum_data.get("ask_iv")
            ask_iv = (float(raw_ask_iv) / 100.0) if raw_ask_iv is not None and float(raw_ask_iv) > 0 else None

            k_fwd = (strike / fwd) if fwd > 0 else 1.0
            log_m = np.log(k_fwd) if fwd > 0 and strike > 0 else 0.0

            obs = OptionMarketObservation(
                provider="DERIBIT",
                underlying="BTC",
                instrument_name=name,
                option_type=opt_type,
                strike=strike,
                expiry_timestamp=expiry_dt,
                maturity_years=tau,
                days_to_expiry=days_to_exp,
                underlying_price=und_price,
                forward_price=fwd,
                log_forward_moneyness=float(log_m),
                forward_moneyness=float(k_fwd),
                mark_price=sum_data.get("mark_price"),
                bid_price=sum_data.get("bid_price"),
                ask_price=sum_data.get("ask_price"),
                mark_iv=mark_iv,
                bid_iv=bid_iv,
                ask_iv=ask_iv,
                open_interest=float(sum_data.get("open_interest") or 0.0),
                volume=float(sum_data.get("volume") or 0.0),
                retrieval_timestamp=retrieval_dt,
                data_timestamp=datetime.fromtimestamp(sum_data.get("creation_timestamp", exp_ts_ms) / 1000.0, tz=timezone.utc) if sum_data.get("creation_timestamp") else None,
            )

            # Apply filtering
            status, reason = self._filter_observation(obs)
            obs.quality_status = status
            obs.exclusion_reason = reason
            observations.append(obs)

        meta = {
            "retrieval_timestamp": retrieval_dt.isoformat(),
            "raw_instruments_count": len(payload.get("instruments", [])),
            "raw_summaries_count": len(payload.get("summaries", [])),
            "spot_or_index_price": spot_est,
            "cache_status": self.cache_status,
        }
        return observations, meta

    def _filter_observation(self, obs: OptionMarketObservation) -> tuple[str, str | None]:
        if obs.maturity_years <= 0 or obs.days_to_expiry < self.config.minimum_days_to_expiry:
            return "EXCLUDED", f"Expiry too short (< {self.config.minimum_days_to_expiry:.0f}d)"
        if obs.days_to_expiry > self.config.maximum_days_to_expiry:
            return "EXCLUDED", f"Expiry too long (> {self.config.maximum_days_to_expiry:.0f}d)"
        if obs.mark_iv is None or not np.isfinite(obs.mark_iv) or obs.mark_iv <= 0:
            return "EXCLUDED", "Missing or non-finite mark IV"
        iv_pct = obs.mark_iv * 100.0
        if iv_pct < self.config.minimum_iv_pct or iv_pct > self.config.maximum_iv_pct:
            return "EXCLUDED", f"IV {iv_pct:.1f}% outside plausibility bounds [{self.config.minimum_iv_pct}, {self.config.maximum_iv_pct}]"
        if obs.forward_price <= 0:
            return "EXCLUDED", "Invalid non-positive forward price"
        if obs.forward_moneyness < self.config.forward_moneyness_min or obs.forward_moneyness > self.config.forward_moneyness_max:
            return "EXCLUDED", f"Moneyness {obs.forward_moneyness:.2f} outside bounds [{self.config.forward_moneyness_min}, {self.config.forward_moneyness_max}]"
        if obs.bid_price is not None and obs.ask_price is not None and obs.bid_price > 0 and obs.ask_price > 0:
            if obs.bid_price > obs.ask_price:
                return "EXCLUDED", "Crossed bid-ask spread"
            spread_rel = (obs.ask_price - obs.bid_price) / max(1e-4, 0.5 * (obs.bid_price + obs.ask_price))
            if spread_rel > self.config.maximum_relative_bid_ask_spread and self.config.require_two_sided_market:
                return "EXCLUDED", f"Excessive relative bid-ask spread ({spread_rel:.2f})"

        return "VALID", None


class RSSNewsProvider:
    def __init__(self, sources: dict) -> None:
        self.feeds = sources.get("feeds", [])
        self.keywords = [k.lower() for k in sources.get("relevance_keywords", [])]

    def fetch_news(self, max_items: int = 5) -> list[NewsItem]:
        scored_items: list[tuple[float, NewsItem]] = []
        for feed in self.feeds:
            if not feed.get("enabled", True):
                continue
            authority = feed.get("authority_weight", 0.5)
            try:
                parsed = feedparser.parse(feed.get("url", ""))
                for e in parsed.entries[:8]:
                    title = html.unescape(e.get("title", "")).strip()
                    summary = html.unescape(e.get("summary", "")).strip()
                    text_to_scan = f"{title} {summary}".lower()
                    keyword_hits = sum(1 for kw in self.keywords if kw in text_to_scan)
                    score = (authority * 2.0) + (keyword_hits * 1.5)
                    # Filter out administrative notices & bank application approvals
                    admin_terms = ["enforcement action", "procurement", "vacancy", "internship", "bank application", "foia", "administrative order"]
                    if any(term in text_to_scan for term in admin_terms):
                        score -= 5.0
                    if score > 0:
                        scored_items.append((score, NewsItem(
                            title=title,
                            publisher=feed.get("name", ""),
                            published_at=datetime.now(timezone.utc),
                            url=e.get("link", "#"),
                            category=feed.get("category", "Central Bank"),
                            relevance_score=score,
                        )))
            except Exception:
                continue
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:max_items]]