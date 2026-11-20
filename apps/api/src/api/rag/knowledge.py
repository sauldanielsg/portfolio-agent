from pydantic import BaseModel
from api.schemas.profile import Profile
import hashlib
import json

def get_profile_version(profile: Profile) -> str:
    canonical_json = json.dumps(
        profile.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()

class KnowledgeChunk(BaseModel):
    id: str
    profile_id: str
    source_id: str
    document_version: str
    text: str

def build_profile_chunks(
    profile: Profile,
    document_version: str,
) -> list[KnowledgeChunk]:
    chunks = [
        KnowledgeChunk(
            id=f"{profile.profileId}:summary",
            profile_id=profile.profileId,
            source_id="summary",
            document_version=document_version,
            text=f"{profile.name}. {profile.headline}. {profile.summary}",
        )
    ]

    for experience in profile.experience:
        context = (
            f"{profile.name} worked as {experience.title} "
            f"at {experience.company}, "
            f"{experience.startDate} to "
            f"{experience.endDate or 'Present'}."
        )

        for index, highlight in enumerate(experience.highlights):
            chunks.append(
                KnowledgeChunk(
                    id=f"{profile.profileId}:experience:{experience.id}:{index}",
                    profile_id=profile.profileId,
                    source_id=f"experience:{experience.id}",
                    document_version=document_version,
                    text=f"{context} {highlight}",
                )
            )

    for group in profile.skills:
        chunks.append(
            KnowledgeChunk(
                id=f"{profile.profileId}:skills:{group.category}",
                profile_id=profile.profileId,
                source_id=f"skills:{group.category}",
                document_version=document_version,
                text=(
                    f"{profile.name}'s listed {group.category} skills: "
                    f"{', '.join(group.items)}."
                ),
            )
        )

    return chunks
