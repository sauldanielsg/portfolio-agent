import pytest

from unittest.mock import MagicMock

from api.rag.knowledge import KnowledgeChunk
from api.rag.retrieval import KnowledgeIndex, cosine_similarity


def test_same_direction():
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1)


def test_perpendicular_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)


def test_opposite_direction():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1)


@pytest.mark.parametrize("left,right", [
    ([], []),
    ([1], [1, 2]),
    ([0, 0], [1, 0]),
    ([float("nan"), 1], [1, 0]),
    ([float("inf"), 1], [1, 0]),
])
def test_invalid_vectors_are_rejected(left, right):
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


def make_chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=chunk_id,
        profile_id="test",
        source_id=f"experience:{chunk_id}",
        document_version="v1",
        text=text,
    )


@pytest.fixture
def index_and_embedder():
    embedder = MagicMock()
    embedder.embed_documents.return_value = [[1.0, 0.0], [0.0, 1.0]]
    embedder.embed_query.return_value = [0.0, 1.0]
    chunks = [
        make_chunk("frontend", "Built React interfaces."),
        make_chunk("data", "Built Python data pipelines."),
    ]
    return KnowledgeIndex(chunks, embedder), embedder


def test_search_ranks_chunks_and_preserves_sources(index_and_embedder):
    index, embedder = index_and_embedder
    results = index.search("Python experience", top_k=1)
    assert len(results) == 1
    assert results[0].chunk == make_chunk("data", "Built Python data pipelines.")
    assert results[0].score == pytest.approx(1.0)
    embedder.embed_documents.assert_called_once_with([
        "Built React interfaces.", "Built Python data pipelines.",
    ])
    embedder.embed_query.assert_called_once_with("Python experience")


def test_repeated_searches_do_not_reembed_documents(index_and_embedder):
    index, embedder = index_and_embedder
    index.search("Python")
    index.search("React")
    embedder.embed_documents.assert_called_once()
    assert embedder.embed_query.call_count == 2


def test_search_returns_available_results_in_descending_order(index_and_embedder):
    index, _ = index_and_embedder
    results = index.search("Python", top_k=10)
    assert [result.chunk.id for result in results] == ["data", "frontend"]
    assert results[0].score >= results[1].score


def test_search_trims_query(index_and_embedder):
    index, embedder = index_and_embedder
    index.search("  Python  ")
    embedder.embed_query.assert_called_once_with("Python")


@pytest.mark.parametrize("query,top_k", [("", 1), ("   ", 1), ("Python", 0), ("Python", -1)])
def test_invalid_search_does_not_embed_query(index_and_embedder, query, top_k):
    index, embedder = index_and_embedder
    with pytest.raises(ValueError):
        index.search(query, top_k)
    embedder.embed_query.assert_not_called()


def test_empty_index_returns_no_matches():
    embedder = MagicMock()
    index = KnowledgeIndex([], embedder)
    assert index.search("Python") == []
    embedder.embed_documents.assert_not_called()
    embedder.embed_query.assert_not_called()


@pytest.mark.parametrize("vectors", [[], [[1, 0], [0, 1]]])
def test_index_rejects_wrong_embedding_count(vectors):
    embedder = MagicMock()
    embedder.embed_documents.return_value = vectors
    with pytest.raises(ValueError, match="one embedding per chunk"):
        KnowledgeIndex([make_chunk("data", "Python pipelines")], embedder)


def test_index_rejects_mixed_dimensions():
    embedder = MagicMock()
    embedder.embed_documents.return_value = [[1, 0], [1, 0, 0]]
    with pytest.raises(ValueError, match="equal dimensions"):
        KnowledgeIndex([make_chunk("a", "A"), make_chunk("b", "B")], embedder)


@pytest.mark.parametrize("vector", [[], [0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_index_rejects_invalid_document_vectors(vector):
    embedder = MagicMock()
    embedder.embed_documents.return_value = [vector]
    with pytest.raises(ValueError):
        KnowledgeIndex([make_chunk("data", "Python pipelines")], embedder)


def test_search_rejects_incompatible_query_dimensions(index_and_embedder):
    index, embedder = index_and_embedder
    embedder.embed_query.return_value = [1, 0, 0]
    with pytest.raises(ValueError, match="equal dimensions"):
        index.search("Python")


def test_hybrid_search_recovers_matching_accomplishment_despite_semantic_bias():
    chunks = [
        KnowledgeChunk(id="skills", profile_id="test", source_id="skills:Languages",
                       document_version="v1", text="Python, Java, React"),
        make_chunk("intuit", "At Intuit, built Python data pipelines."),
        make_chunk("collins", "At Collins, built AWS infrastructure."),
    ]
    embedder = MagicMock()
    embedder.embed_query.return_value = [1, 0]
    # Reproduce the failure: semantic similarity strongly favors the skill list.
    index = KnowledgeIndex(chunks, embedder, vectors=[[1, 0], [0.4, 0.9], [0.3, 0.9]])
    assert index.search("what python experience?", top_k=1)[0].chunk.id == "intuit"


@pytest.mark.parametrize("query", ["Kubernetes experience", "Kubernetes"])
def test_unknown_technology_experience_does_not_return_unrelated_evidence(index_and_embedder, query):
    index, _ = index_and_embedder
    assert index.search(query) == []


def test_keyword_matching_does_not_confuse_c_with_cpp():
    chunks = [make_chunk("c", "Built tools in C."), make_chunk("cpp", "Built tools in C++.")]
    embedder = MagicMock()
    embedder.embed_query.return_value = [1, 0]
    index = KnowledgeIndex(chunks, embedder, vectors=[[1, 0], [0.5, 0.8]])
    assert index.search("C++ experience", top_k=1)[0].chunk.id == "cpp"
