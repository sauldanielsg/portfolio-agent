from unittest.mock import MagicMock

import pytest

from agent_middleware import DailyLimitReached, create_quota_middleware
from quota import Quota


def test_middleware_reserves_before_calling_model():
    quota = Quota(1, "America/Los_Angeles")
    middleware = create_quota_middleware(quota)
    request = MagicMock()
    expected_response = MagicMock()

    def handler(received_request):
        assert received_request is request
        assert quota.used == 1
        return expected_response

    result = middleware.wrap_model_call(request, handler)

    assert result is expected_response


def test_exhausted_quota_does_not_call_model():
    quota = Quota(0, "America/Los_Angeles")
    middleware = create_quota_middleware(quota)
    handler = MagicMock()

    with pytest.raises(DailyLimitReached):
        middleware.wrap_model_call(MagicMock(), handler)

    handler.assert_not_called()


def test_failed_model_call_still_consumes_reservation():
    quota = Quota(1, "America/Los_Angeles")
    middleware = create_quota_middleware(quota)
    handler = MagicMock(side_effect=RuntimeError("Model failed"))

    with pytest.raises(RuntimeError, match="Model failed"):
        middleware.wrap_model_call(MagicMock(), handler)

    assert quota.used == 1

    with pytest.raises(DailyLimitReached):
        middleware.wrap_model_call(MagicMock(), handler)

    handler.assert_called_once()