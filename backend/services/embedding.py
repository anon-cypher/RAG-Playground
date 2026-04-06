"""Embedding service — OpenRouter/OpenAI wrapper with batching."""
import numpy as np
from typing import Optional, Callable
from openai import OpenAI

import config


def embed_texts(
    texts: list[str],
    batch_size: int | None = None,
    on_batch_progress: Optional[Callable[[int, int], None]] = None
) -> np.ndarray:
    """Embed a list of texts into vectors using OpenRouter API.

    Args:
        texts: List of strings to embed.
        batch_size: Batch size for encoding. Defaults to config value.

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    # Note: importing here to prevent circular imports if pipeline Service imports embedding
    from services.pipeline import get_config
    cfg = get_config()

    if not texts:
        return np.array([]).reshape(0, cfg.embedding_dim)

    # Note: importing here to prevent circular imports if pipeline Service imports embedding
    from services.pipeline import get_config
    cfg = get_config()
    api_key = cfg.openrouter_api_key

    if not api_key:
        raise ValueError("OpenRouter API Key is now required for embeddings (local models are disabled to save memory). Please add it in Pipeline Config.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    bs = batch_size or config.EMBEDDING_BATCH_SIZE
    all_embeddings = []
    total_batches = (len(texts) + bs - 1) // bs
    
    for batch_idx, i in enumerate(range(0, len(texts), bs)):
        batch = texts[i:i+bs]
        try:
            response = client.embeddings.create(
                input=batch,
                model=cfg.embedding_model
            )
            batch_embeds = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeds)
            if on_batch_progress:
                on_batch_progress(batch_idx + 1, total_batches)
        except Exception as e:
            raise RuntimeError(f"Embedding API failed: {str(e)}")
            
    return np.array(all_embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Returns:
        numpy array of shape (1, embedding_dim)
    """
    return embed_texts([query])


def get_embedding_dim() -> int:
    """Get the embedding dimension."""
    from services.pipeline import get_config
    return get_config().embedding_dim
