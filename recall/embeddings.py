import numpy as np
from sentence_transformers import SentenceTransformer

from recall import settings as settings_module

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings_module.settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed documents/chunks for storage."""
    model = _get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=512,
        show_progress_bar=len(texts) > 100,
    )
    return [embeddings[i] for i in range(len(texts))]


def embed_query(text: str) -> np.ndarray:
    """Embed a search query."""
    model = _get_model()
    embeddings: np.ndarray = model.encode(
        [f"Represent this sentence for searching relevant passages: {text}"],
        normalize_embeddings=True,
    )
    return embeddings[0]
