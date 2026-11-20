import logging

from api.core.paths import PROFILE_PATH, PROJECT_ROOT
from api.core.settings import Settings
from api.schemas.profile import Profile
from api.usage.quota import Quota
from api.providers.gemini import GeminiProvider
from api.agents.portfolio import PortfolioAgent
from api.rag.knowledge import get_profile_version
from api.rag.retrieval import KnowledgeIndex

settings = Settings()
logger = logging.getLogger("uvicorn.error")
quota = Quota(settings.daily_model_call_limit, settings.quota_timezone)
provider = GeminiProvider(settings)

profile = Profile.model_validate_json(
    PROFILE_PATH.read_text(encoding="utf-8")
)

knowledge_index: KnowledgeIndex | None = None

if settings.rag_enabled:
    if (
        settings.ai_mode != "gemini"
        or settings.agent_backend != "langchain"
    ):
        raise ValueError(
            "RAG requires AI_MODE=gemini and AGENT_BACKEND=langchain"
        )

    from api.rag.local_embedder import LocalEmbedder

    embedder = LocalEmbedder()

    index_path = (
        PROJECT_ROOT
        / "artifacts"
        / profile.profileId
        / "index.json"
    )

    knowledge_index = KnowledgeIndex.load(
        index_path,
        embedder,
        profile_id=profile.profileId,
        document_version=get_profile_version(profile),
        embedding_model=embedder.model_id,
    )

portfolio_agent = (
    PortfolioAgent(
        settings,
        profile,
        quota,
        index=knowledge_index,
    )
    if settings.ai_mode == "gemini"
    and settings.agent_backend == "langchain"
    else None
)
