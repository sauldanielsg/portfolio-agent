from fastapi import FastAPI, HTTPException
from pathlib import Path
import logging
from models import Profile, ChatRequest, ChatResponse
from settings import Settings
from quota import Quota
from gemini_provider import GeminiProvider, ProviderError, ProviderRateLimitError
from portfolio_agent import PortfolioAgent
from agent_middleware import DailyLimitReached

PROFILE_PATH = (
    Path(__file__).resolve().parents[3]
    / "profiles"
    / "default"
    / "profile.json"
)

settings = Settings()
logger = logging.getLogger("uvicorn.error")
quota = Quota(settings.daily_model_call_limit, settings.quota_timezone)
provider = GeminiProvider(settings)
app = FastAPI(title="AI Portfolio API")

profile = Profile.model_validate_json(
    PROFILE_PATH.read_text(encoding="utf-8")
)

portfolio_agent = (
    PortfolioAgent(settings, profile, quota)
    if settings.ai_mode == "gemini"
    and settings.agent_backend == "langchain"
    else None
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not settings.ai_enabled:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "AI_UNAVAILABLE",
                "message": "I'm taking a break. Please check back later.",
            },
        )
    if settings.ai_mode == "gemini" and settings.agent_backend == "langchain":
        if portfolio_agent is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "AI_UNAVAILABLE",
                    "message": "The assistant is unavailable. Please try again later.",
                },
            )

        try:
            return portfolio_agent.answer(
                request.message,
                request.history,
            )
        except DailyLimitReached as error:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "DAILY_LIMIT_REACHED",
                    "message": (
                        "The AI assistant has reached its daily limit. "
                        "Please try again tomorrow."
                    ),
                },
            ) from error
        except Exception as error:
            logger.warning(
                "Portfolio agent failed: category=%s",
                type(error).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "AI_UNAVAILABLE",
                    "message": "I'm taking a break. Please check back later.",
                },
            ) from error
    if settings.ai_mode == "gemini":
        if not quota.reserve():
            raise HTTPException(status_code=429, detail={
                "error": "DAILY_LIMIT_REACHED",
                "message": "The AI assistant has reached its daily limit. Please try again tomorrow.",
            })
        try:
            return provider.answer(
                request.message, 
                profile, 
                request.history,
            )
        except ProviderRateLimitError as error:
            raise HTTPException(status_code=503, detail={
                "error": "AI_UNAVAILABLE",
                "message": "The AI service's usage limit has been reached. Please try again later.",
            }) from error
        except ProviderError as error:
            cause = error.__cause__
            # Log diagnostic categories only, never keys, prompts, or provider bodies.
            logger.warning(
                "Gemini request failed: category=%s cause=%s status=%s",
                str(error),
                type(cause).__name__ if cause else "none",
                getattr(cause, "code", "none"),
            )
            raise HTTPException(status_code=503, detail={
                "error": "AI_UNAVAILABLE",
                "message": "I'm taking a break. Please check back later.",
            }) from error

    if "resume" in request.message.lower():
        return ChatResponse(
            message="Here is my full-stack resume.",
            action="show_resume",
        )
    return ChatResponse(
        message="Ask me about my full-stack experience, or ask to see my resume."
    )
    
@app.get("/api/v1/profile", response_model=Profile)
def get_profile() -> Profile:
    return profile
