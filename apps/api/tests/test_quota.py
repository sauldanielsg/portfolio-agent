from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import quota as quota_module
from quota import Quota


def freeze_time(monkeypatch, instant):
    """Control time without sleeping or depending on the test machine's date."""
    class FrozenDateTime:
        @classmethod
        def now(cls, tz):
            return instant.astimezone(tz)

    monkeypatch.setattr(quota_module, "datetime", FrozenDateTime)


def test_reservations_stop_at_daily_limit():
    quota = Quota(2, "America/Los_Angeles")
    assert [quota.reserve() for _ in range(4)] == [True, True, False, False]
    assert quota.used == 2


def test_zero_limit_blocks_all_calls():
    quota = Quota(0, "America/Los_Angeles")
    assert quota.reserve() is False
    assert quota.used == 0


def test_limit_resets_at_local_midnight(monkeypatch):
    # In November, LA midnight is 08:00 UTC. UTC midnight must not reset it.
    freeze_time(monkeypatch, datetime(2026, 11, 17, 23, 59, tzinfo=timezone.utc))
    quota = Quota(1, "America/Los_Angeles")
    assert quota.reserve() is True

    freeze_time(monkeypatch, datetime(2026, 11, 18, 0, 1, tzinfo=timezone.utc))
    assert quota.reserve() is False
    freeze_time(monkeypatch, datetime(2026, 11, 18, 7, 59, tzinfo=timezone.utc))
    assert quota.reserve() is False

    freeze_time(monkeypatch, datetime(2026, 11, 18, 8, 0, tzinfo=timezone.utc))
    assert quota.reserve() is True
    assert quota.current_date.isoformat() == "2026-11-18"
    # A second call on the new day must not reset the counter again.
    assert quota.reserve() is False
    assert quota.used == 1


def test_concurrent_reservations_cannot_exceed_limit():
    quota = Quota(5, "America/Los_Angeles")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: quota.reserve(), range(100)))
    assert sum(results) == 5
    assert quota.used == 5
