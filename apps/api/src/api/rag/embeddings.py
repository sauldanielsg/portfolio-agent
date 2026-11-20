from typing import Protocol

class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per document, in the same order."""

    def embed_query(self, text: str) -> list[float]:
        """Return an embedding vector for a search query."""
