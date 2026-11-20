
from api.rag.knowledge import build_profile_chunks, get_profile_version
from api.rag.local_embedder import LocalEmbedder
from api.schemas.profile import Profile
from api.rag.retrieval import KnowledgeIndex


from api.core.paths import PROJECT_ROOT


def main():
    profile_path = PROJECT_ROOT / "profiles/default/profile.json"
    output_path = PROJECT_ROOT / "artifacts/default/index.json"

    profile = Profile.model_validate_json(
        profile_path.read_text(encoding="utf-8")
    )
    version = get_profile_version(profile)
    chunks = build_profile_chunks(profile, version)

    embedder = LocalEmbedder()
    index = KnowledgeIndex(chunks, embedder)

    index.save(
        output_path,
        profile_id=profile.profileId,
        document_version=version,
        embedding_model=embedder.model_id,
    )

    print(f"Saved {len(chunks)} chunks to {output_path}")
    print(f"Document version: {version}")


if __name__ == "__main__":
    main()
