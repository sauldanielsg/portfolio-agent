from fastapi import APIRouter

from api.core import runtime
from api.schemas.profile import Profile

router = APIRouter(prefix="/api/v1", tags=["profile"])


@router.get("/profile", response_model=Profile)
def get_profile() -> Profile:
    return runtime.profile
