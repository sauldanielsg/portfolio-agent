from fastapi import APIRouter, HTTPException

from api.core import runtime
from api.agents.middleware import DailyLimitReached
from api.providers.gemini import ProviderError, ProviderRateLimitError
from api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not runtime.settings.ai_enabled:
        raise HTTPException(status_code=503, detail={'error': 'AI_UNAVAILABLE', 'message': "I'm taking a break. Please check back later."})
    if runtime.settings.ai_mode == 'gemini' and runtime.settings.agent_backend == 'langchain':
        if runtime.portfolio_agent is None:
            raise HTTPException(status_code=503, detail={'error': 'AI_UNAVAILABLE', 'message': 'The assistant is unavailable. Please try again later.'})
        try:
            return runtime.portfolio_agent.answer(request.message, request.history)
        except DailyLimitReached as error:
            raise HTTPException(status_code=429, detail={'error': 'DAILY_LIMIT_REACHED', 'message': 'The AI assistant has reached its daily limit. Please try again tomorrow.'}) from error
        except Exception as error:
            runtime.logger.warning('Portfolio agent failed: category=%s', type(error).__name__)
            raise HTTPException(status_code=503, detail={'error': 'AI_UNAVAILABLE', 'message': "I'm taking a break. Please check back later."}) from error
    if runtime.settings.ai_mode == 'gemini':
        if not runtime.quota.reserve():
            raise HTTPException(status_code=429, detail={'error': 'DAILY_LIMIT_REACHED', 'message': 'The AI assistant has reached its daily limit. Please try again tomorrow.'})
        try:
            return runtime.provider.answer(request.message, runtime.profile, request.history)
        except ProviderRateLimitError as error:
            raise HTTPException(status_code=503, detail={'error': 'AI_UNAVAILABLE', 'message': "The AI service's usage limit has been reached. Please try again later."}) from error
        except ProviderError as error:
            cause = error.__cause__
            runtime.logger.warning('Gemini request failed: category=%s cause=%s status=%s', str(error), type(cause).__name__ if cause else 'none', getattr(cause, 'code', 'none'))
            raise HTTPException(status_code=503, detail={'error': 'AI_UNAVAILABLE', 'message': "I'm taking a break. Please check back later."}) from error
    if 'resume' in request.message.lower():
        return ChatResponse(message='Here is my full-stack resume.', action='show_resume')
    return ChatResponse(message='Ask me about my full-stack experience, or ask to see my resume.')
