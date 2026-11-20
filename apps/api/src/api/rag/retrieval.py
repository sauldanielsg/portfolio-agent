from math import hypot, isfinite
from pathlib import Path
from pydantic import BaseModel
from typing import Literal
import re

from api.rag.embeddings import Embedder
from api.rag.knowledge import KnowledgeChunk


# Remove question scaffolding, not technology names. Keep punctuation in C++,
# C#, Node.js, etc. so exact matching does not collapse distinct technologies.
QUERY_STOP_WORDS = frozenset("""
a an the what which who where when how does do did has have had is are was were
can could would you me my his her their they he she it its about at in on of to
for and or with as from tell show please experience experiences experienced
skill skills project projects role roles work worked working background use used
uses using knowledge technology technologies programming language languages
""".split())


def search_terms(text: str) -> set[str]:
    return set(re.findall(r"[\w]+(?:[.+#-][\w]+)*(?:\+\+|#)?", text.casefold()))

class IndexedChunk(BaseModel):
    chunk: KnowledgeChunk
    vector: list[float]

class IndexArtifact(BaseModel):
    schema_version: Literal[1] = 1
    profile_id: str
    document_version: str
    embedding_model: str
    entries: list[IndexedChunk]

def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Vectors must be nonempty and have equal dimensions")
    if not all(isfinite(value) for value in left + right):
        raise ValueError("Vectors must contain only finite numbers")

    left_norm = hypot(*left)
    right_norm = hypot(*right)

    if left_norm == 0 or right_norm == 0:
        raise ValueError("Vectors must have nonzero magnitude")

    return sum(
        (a / left_norm) * (b / right_norm)
        for a, b in zip(left, right)
    )

class SearchResult(BaseModel):
    chunk: KnowledgeChunk
    # Hybrid ranking score, not a confidence probability or raw cosine score.
    score: float

class KnowledgeIndex:
    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        embedder: Embedder,
        *,
        vectors: list[list[float]] | None = None,
    ):
        self.chunks = list(chunks)
        self.embedder = embedder

        if vectors is not None:
            self.vectors = [list(vector) for vector in vectors]
        else:
            self.vectors = (
                embedder.embed_documents([chunk.text for chunk in self.chunks])
                if self.chunks
                else []
            )

        if len(self.vectors) != len(self.chunks):
            raise ValueError("Expected one embedding per chunk")

        dimensions = {len(vector) for vector in self.vectors}
        if len(dimensions) > 1:
            raise ValueError("All embeddings must have equal dimensions")

        for vector in self.vectors:
            cosine_similarity(vector, vector)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query = query.strip()

        if not query:
            raise ValueError("Search query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.chunks:
            return []

        query_vector = self.embedder.embed_query(query)
        chunk_terms = [search_terms(chunk.text) for chunk in self.chunks]
        # Repeated context (such as the owner's name) should not dominate matches.
        common_terms = set.intersection(*chunk_terms) if chunk_terms else set()
        terms = search_terms(query) - QUERY_STOP_WORDS - common_terms
        work_question = bool(search_terms(query) & {"experience", "experiences", "work", "worked", "role", "roles"})

        # A single-term lookup or short explicit experience query about an unlisted term should not
        # return unrelated work as evidence. Broader/paraphrased queries retain
        # semantic retrieval. This is a conservative heuristic, not proof of absence.
        explicit_lookup = work_question or len(terms) == 1
        if explicit_lookup and 0 < len(terms) <= 2 and not any(terms & words for words in chunk_terms):
            return []

        results = []
        for chunk, vector, words in zip(self.chunks, self.vectors, chunk_terms):
            semantic = cosine_similarity(query_vector, vector)
            if terms:
                coverage = len(terms & words) / len(terms)
                score = 0.5 * semantic + 0.5 * coverage
                if work_question or len(terms) == 1:
                    # Concrete matching accomplishments deserve representation
                    # ahead of generic skill lists for experience questions.
                    work_match = coverage == 1 and chunk.source_id.startswith("experience:")
                    score = 0.3 * semantic + 0.4 * coverage + 0.3 * work_match
            else:
                score = semantic
            results.append(SearchResult(chunk=chunk, score=score))

        return sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )[:top_k]

    def save(
        self,
        path: Path,
        profile_id: str,
        document_version: str,
        embedding_model: str,
    ) -> None:
        if not embedding_model.strip():
            raise ValueError("Embedding model must not be empty")

        for chunk in self.chunks:
            if chunk.profile_id != profile_id:
                raise ValueError("Chunk belongs to a different profile")
            if chunk.document_version != document_version:
                raise ValueError("Chunk has a different document version")

        artifact = IndexArtifact(
            profile_id=profile_id,
            document_version=document_version,
            embedding_model=embedding_model,
            entries=[
                IndexedChunk(chunk=chunk, vector=vector)
                for chunk, vector in zip(
                    self.chunks,
                    self.vectors,
                    strict=True,
                )
            ]
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            artifact.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Path,
        embedder: Embedder,
        *,
        profile_id: str,
        document_version: str,
        embedding_model: str,
    ) -> "KnowledgeIndex":
        artifact = IndexArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )

        if artifact.profile_id != profile_id:
            raise ValueError("Index belongs to a different profile")
        if artifact.document_version != document_version:
            raise ValueError("Index has a different document version")
        if artifact.embedding_model != embedding_model:
            raise ValueError("Index uses a different embedding model")

        for entry in artifact.entries:
            if entry.chunk.profile_id != artifact.profile_id:
                raise ValueError("Chunk profile does not match index")
            if entry.chunk.document_version != artifact.document_version:
                raise ValueError("Chunk version does not match index")

        return cls(
            chunks=[entry.chunk for entry in artifact.entries],
            embedder=embedder,
            vectors=[entry.vector for entry in artifact.entries],
        )
