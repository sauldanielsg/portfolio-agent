import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from api.rag.knowledge import KnowledgeChunk
from api.rag.retrieval import KnowledgeIndex


@pytest.fixture
def saved_index(tmp_path):
    chunks = [
        KnowledgeChunk(
            id=name, profile_id="test", source_id=f"experience:{name}",
            document_version="v1", text=text,
        )
        for name, text in [("frontend", "React interfaces"), ("data", "Python pipelines")]
    ]
    index = KnowledgeIndex(chunks, MagicMock(), vectors=[[1, 0], [0, 1]])
    path = tmp_path / "nested" / "index.json"
    index.save(path, "test", "v1", "fake-model")
    return path, index


def load(path, embedder):
    return KnowledgeIndex.load(
        path, embedder, profile_id="test", document_version="v1",
        embedding_model="fake-model",
    )


def test_saved_index_restores_search_without_reembedding(saved_index):
    path, original = saved_index
    embedder = MagicMock()
    embedder.embed_query.return_value = [0, 1]
    restored = load(path, embedder)
    assert restored.chunks == original.chunks
    assert restored.vectors == original.vectors
    results = restored.search("Python", top_k=1)
    assert results[0].chunk.id == "data"
    assert results[0].chunk.source_id == "experience:data"
    assert results[0].chunk.document_version == "v1"
    assert results[0].score == pytest.approx(1)
    embedder.embed_documents.assert_not_called()
    embedder.embed_query.assert_called_once_with("Python")


def test_empty_index_round_trip(tmp_path):
    embedder = MagicMock()
    path = tmp_path / "empty.json"
    index = KnowledgeIndex([], embedder)
    index.save(path, "test", "v1", "fake-model")
    restored = load(path, embedder)
    assert restored.chunks == restored.vectors == []
    assert restored.search("Python") == []
    embedder.embed_documents.assert_not_called()
    embedder.embed_query.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("profile_id", "other"), ("document_version", "v2"),
    ("embedding_model", "other-model"),
])
def test_load_rejects_incompatible_metadata(saved_index, field, value):
    path, _ = saved_index
    expected = dict(profile_id="test", document_version="v1", embedding_model="fake-model")
    expected[field] = value
    embedder = MagicMock()
    with pytest.raises(ValueError):
        KnowledgeIndex.load(path, embedder, **expected)
    embedder.embed_documents.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("profile_id", "other"), ("document_version", "v2"),
])
def test_load_validates_metadata_of_every_entry(saved_index, field, value):
    path, _ = saved_index
    artifact = json.loads(path.read_text())
    # Corrupt the last entry to catch an accidental return inside the loop.
    artifact["entries"][-1]["chunk"][field] = value
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="Chunk"):
        load(path, MagicMock())


@pytest.mark.parametrize("vector", [[], [0, 0], [1, 0, 0], [float("nan"), 1]])
def test_load_rejects_invalid_saved_vectors(saved_index, vector):
    path, _ = saved_index
    artifact = json.loads(path.read_text())
    artifact["entries"][-1]["vector"] = vector
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        load(path, MagicMock())


@pytest.mark.parametrize("content", ["not JSON", '{"schema_version": 1}'])
def test_load_rejects_malformed_artifact(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content)
    with pytest.raises(ValidationError):
        load(path, MagicMock())


def test_load_rejects_unsupported_schema_version(saved_index):
    path, _ = saved_index
    artifact = json.loads(path.read_text())
    artifact["schema_version"] = 2
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValidationError):
        load(path, MagicMock())


@pytest.mark.parametrize("profile_id,version,model", [
    ("other", "v1", "fake-model"), ("test", "v2", "fake-model"),
    ("test", "v1", "   "),
])
def test_invalid_save_does_not_overwrite_existing_file(saved_index, profile_id, version, model):
    path, index = saved_index
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError):
        index.save(path, profile_id, version, model)
    assert path.read_bytes() == original_bytes


def test_save_rejects_mismatched_chunk_vector_count(saved_index):
    path, index = saved_index
    original_bytes = path.read_bytes()
    index.vectors.pop()
    with pytest.raises(ValueError):
        index.save(path, "test", "v1", "fake-model")
    assert path.read_bytes() == original_bytes


def test_republished_index_replaces_outdated_searchable_content(saved_index):
    path, old_index = saved_index
    updated_chunk = old_index.chunks[1].model_copy(update={
        "text": "Java backend services", "document_version": "v2",
    })
    rebuilt = KnowledgeIndex([updated_chunk], MagicMock(), vectors=[[0, 1]])
    rebuilt.save(path, "test", "v2", "fake-model")
    embedder = MagicMock()
    embedder.embed_query.return_value = [0, 1]
    restored = KnowledgeIndex.load(
        path, embedder, profile_id="test", document_version="v2", embedding_model="fake-model",
    )
    results = restored.search("backend services")
    assert len(results) == 1
    assert results[0].chunk.text == "Java backend services"
    assert results[0].chunk.document_version == "v2"
    assert all("Python" not in chunk.text for chunk in restored.chunks)
    with pytest.raises(ValueError, match="document version"):
        load(path, embedder)
    embedder.embed_documents.assert_not_called()
