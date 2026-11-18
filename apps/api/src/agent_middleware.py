from langchain.agents.middleware import wrap_model_call

from quota import Quota

class DailyLimitReached(Exception):
    """No model calls remain for the current day."""
    
def create_quota_middleware(quota: Quota):
    @wrap_model_call
    def enforce_quota(request, handler):
        if not quota.reserve():
            raise DailyLimitReached(
                "The AI assistant has reached its daily limit."
            )
        return handler(request)
    return enforce_quota
