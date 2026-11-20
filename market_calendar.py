"""Trading calendar service with TARGET settlement checks."""

from datetime import date, timedelta
import pandas as pd
import pandas_market_calendars as mcal


class MarketCalendarService:
    def __init__(self) -> None:
        try:
            self.target_cal = mcal.get_calendar("TARGET")
        except Exception:
            self.target_cal = None

    def get_previous_business_day(self, ref_date: date) -> date:
        start_date = ref_date - timedelta(days=14)
        end_date = ref_date - timedelta(days=1)
        if self.target_cal is not None:
            sched = self.target_cal.schedule(start_date=start_date.isoformat(), end_date=end_date.isoformat())
            if not sched.empty:
                return pd.to_datetime(sched.index[-1]).date()
        # Standard weekday rollback fallback
        cur = ref_date - timedelta(days=1)
        while cur.weekday() >= 5:
            cur -= timedelta(days=1)
        return cur

    def is_trading_day(self, check_date: date) -> bool:
        if check_date.weekday() >= 5:
            return False
        if self.target_cal is not None:
            return not self.target_cal.schedule(start_date=check_date.isoformat(), end_date=check_date.isoformat()).empty
        return True