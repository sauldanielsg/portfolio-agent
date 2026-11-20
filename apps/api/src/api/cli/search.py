from api.core.paths import PROJECT_ROOT
from api.rag.knowledge import get_profile_version
from api.rag.local_embedder import LocalEmbedder
from api.schemas.profile import Profile
from api.rag.retrieval import KnowledgeIndex


def main():
    profile = Profile.model_validate_json(
        (PROJECT_ROOT / "profiles/default/profile.json").read_text(
            encoding="utf-8"
        )
    )
    embedder = LocalEmbedder()

    index = KnowledgeIndex.load(
        PROJECT_ROOT / "artifacts/default/index.json",
        embedder,
        profile_id=profile.profileId,
        document_version=get_profile_version(profile),
        embedding_model=embedder.model_id,
    )

    results = index.search(
        "What Python data pipeline experience does Saul have?",
        top_k=3,
    )

    for result in results:
        print(f"\nScore: {result.score:.3f}")
        print(f"Source: {result.chunk.source_id}")
        print(result.chunk.text)


if __name__ == "__main__":
    main()
