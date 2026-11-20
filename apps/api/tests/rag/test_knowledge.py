from api.rag.knowledge import build_profile_chunks
from api.schemas.profile import Experience, Profile


def make_profile():
    return Profile(
        profileId="test",
        name="Example Engineer",
        headline="Full-Stack Engineer",
        summary="Builds web applications.",
        experience=[
            Experience(
                id="example",
                company="Example Co",
                title="Engineer",
                startDate="2024-01",
                endDate=None,
                highlights=["Built Python data pipelines."],
            )
        ],
        skills=[],
        education=[],
        languages=[],
    )


def test_experience_chunk_preserves_context_and_source():
    chunks = build_profile_chunks(make_profile(), "v1")
    chunk = next(
        chunk for chunk in chunks
        if chunk.source_id == "experience:example"
    )

    assert "Engineer at Example Co" in chunk.text
    assert "2024-01 to Present" in chunk.text
    assert "Built Python data pipelines." in chunk.text
    assert chunk.profile_id == "test"
    assert chunk.document_version == "v1"
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_rebuilding_uses_updated_content():
    profile = make_profile()
    old_chunks = build_profile_chunks(profile, "v1")

    profile.experience[0].highlights = ["Built Java backend services."]
    new_chunks = build_profile_chunks(profile, "v2")

    assert any("Python data pipelines" in chunk.text for chunk in old_chunks)
    assert not any("Python data pipelines" in chunk.text for chunk in new_chunks)
    assert any("Java backend services" in chunk.text for chunk in new_chunks)
    assert all(chunk.document_version == "v2" for chunk in new_chunks)
