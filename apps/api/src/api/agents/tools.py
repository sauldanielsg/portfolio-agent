from langchain.tools import tool
from api.schemas.profile import Profile
from api.rag.retrieval import KnowledgeIndex

def create_portfolio_tools(
    profile: Profile,
    index: KnowledgeIndex | None = None,
):
    @tool
    def get_profile() -> dict:
        """Get verified professional background, experience, and skills."""
        return profile.model_dump()

    @tool
    def get_resume() -> dict:
        """Request the full resume view when the visitor asks to see it."""
        return {
            "profileId": profile.profileId,
            "action": "show_resume",
        }

    tools = [get_profile, get_resume]

    if index is not None:
        @tool
        def search_knowledge(query: str) -> list[dict]:
            """Search verified experience and skills using a specific question."""
            results = index.search(query, top_k=3)
            # Temporary local diagnostic; remove after investigating retrieval.
            print(
                "knowledge search:",
                repr(query),
                [result.chunk.id for result in results],
                flush=True,
            )
            return [
                {
                    "chunkId": result.chunk.id,
                    "sourceId": result.chunk.source_id,
                    "documentVersion": result.chunk.document_version,
                    "text": result.chunk.text,
                    "score": result.score,
                }
                for result in results
            ]

        tools.append(search_knowledge)

    return tools
