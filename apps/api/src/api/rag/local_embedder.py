from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.model = SentenceTransformer(self.model_id, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
