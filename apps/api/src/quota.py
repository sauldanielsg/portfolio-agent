from datetime import datetime
from threading import Lock
from zoneinfo import ZoneInfo

class Quota:
    def __init__(self, daily_limit: int, timezone: str):
        self.daily_limit = daily_limit
        self.timezone = ZoneInfo(timezone)
        self.current_date = datetime.now(self.timezone).date()
        self.used = 0
        self.lock = Lock()
    
    def reserve(self) -> bool:
        """Reserve one call, returning False when today's limit is reached."""
        with self.lock:
            today = datetime.now(self.timezone).date()
            if self.current_date != today:
                self.used = 0
                self.current_date = today
            if self.used >= self.daily_limit:
                return False
            self.used += 1
            return True