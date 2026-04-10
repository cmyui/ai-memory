import numpy as np
from sentence_transformers import SentenceTransformer

from recall import settings as settings_module

_model: SentenceTransformer | None = None
_model_name: str | None = None

# Query/document prefixes per model family
_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge": "Represent this sentence for searching relevant passages: ",
    "nomic-ai/nomic": "search_query: ",
}
_DOC_PREFIXES: dict[str, str] = {
    "nomic-ai/nomic": "search_document: ",
}


def _get_prefix(prefixes: dict[str, str]) -> str:
    name = _model_name or settings_module.settings.embedding_model
    for key, prefix in prefixes.items():
        if key in name:
            return prefix
    return ""


def set_model(model_name: str, truncate_dim: int | None = None) -> None:
    """Override the embedding model. Forces reload."""
    global _model, _model_name
    _model_name = model_name
    kwargs: dict = {}
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    if "nomic" in model_name:
        kwargs["trust_remote_code"] = True
    _model = SentenceTransformer(model_name, **kwargs)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings_module.settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed documents/chunks for storage."""
    model = _get_model()
    prefix = _get_prefix(_DOC_PREFIXES)
    prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts
    embeddings: np.ndarray = model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=512,
        show_progress_bar=len(texts) > 100,
    )
    return [embeddings[i] for i in range(len(texts))]


def embed_query(text: str) -> np.ndarray:
    """Embed a search query."""
    model = _get_model()
    prefix = _get_prefix(_QUERY_PREFIXES)
    embeddings: np.ndarray = model.encode(
        [f"{prefix}{text}"],
        normalize_embeddings=True,
    )
    return embeddings[0]
