"""Retrieval service — FAISS-based vector search with Flat and HNSW indices."""
import numpy as np
import faiss
from typing import Optional
from abc import ABC, abstractmethod

import config


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.is_built = False

    @abstractmethod
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build the index from embeddings."""
        pass

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the index.

        Args:
            query_vec: Query vector of shape (1, dim)
            top_k: Number of results to return

        Returns:
            Tuple of (distances, indices) each of shape (1, top_k)
        """
        pass

    def add_vectors(self, embeddings: np.ndarray) -> None:
        """Add new vectors to an existing index."""
        if self.index is not None:
            self.index.add(embeddings)

    @property
    def total_vectors(self) -> int:
        """Get total number of indexed vectors."""
        return self.index.ntotal if self.index else 0


class FlatRetriever(BaseRetriever):
    """Exact L2 search using IndexFlatL2."""

    def build_index(self, embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.is_built = True

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_built or self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)
        return distances, indices


class HNSWRetriever(BaseRetriever):
    """Graph-based approximate search using IndexHNSWFlat."""

    def __init__(self, m: int = None, ef_construction: int = None, ef_search: int = None):
        super().__init__()
        self.m = m or config.HNSW_M
        self.ef_construction = ef_construction or config.HNSW_EF_CONSTRUCTION
        self.ef_search = ef_search or config.HNSW_EF_SEARCH

    def build_index(self, embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, self.m)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self.index.add(embeddings)
        self.is_built = True

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_built or self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        # Update efSearch dynamically
        self.index.hnsw.efSearch = max(self.ef_search, top_k)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)
        return distances, indices

class IVFRetriever(BaseRetriever):
    """Voronoi cell based partitioning using IndexIVFFlat."""

    def __init__(self, nlist: int = 10):
        super().__init__()
        self.nlist = nlist

    def build_index(self, embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        n = embeddings.shape[0]
        # FAISS needs nlist <= n (ideally nlist << n). If too few points, scale it down.
        actual_nlist = min(self.nlist, max(1, n // 5))
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist, faiss.METRIC_L2)
        
        # Train and add
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        self.is_built = True

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_built or self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        self.index.nprobe = min(self.index.nlist, max(1, self.index.nlist // 2))
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)
        return distances, indices


class PQRetriever(BaseRetriever):
    """Product Quantization for high memory compression using IndexPQ."""

    def __init__(self, m: int = 8, nbits: int = 8):
        super().__init__()
        self.m = m
        self.nbits = nbits

    def build_index(self, embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        n = embeddings.shape[0]
        
        # Product Quantizer requires dim to be a multiple of m
        # And requires enough training points: typically at least 256 for nbits=8
        # If not enough points or invalid dim, we fallback to flat 
        if dim % self.m != 0 or n < 256:
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = faiss.IndexPQ(dim, self.m, self.nbits)
            if not self.index.is_trained:
                self.index.train(embeddings)
                
        self.index.add(embeddings)
        self.is_built = True

    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_built or self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)
        return distances, indices


# Global retriever registry
_retrievers: dict[str, BaseRetriever] = {}
_embeddings_store: Optional[np.ndarray] = None


def get_or_create_retriever(
    index_type: str = "flat",
    hnsw_m: int = None,
    hnsw_ef_construction: int = None,
    hnsw_ef_search: int = None,
) -> BaseRetriever:
    """Get or create a retriever by index type."""
    key = index_type.lower()

    if key == "flat":
        if "flat" not in _retrievers:
            _retrievers["flat"] = FlatRetriever()
        return _retrievers["flat"]
    elif key == "hnsw":
        if "hnsw" not in _retrievers:
            _retrievers["hnsw"] = HNSWRetriever(
                m=hnsw_m,
                ef_construction=hnsw_ef_construction,
                ef_search=hnsw_ef_search,
            )
        return _retrievers["hnsw"]
    elif key == "ivf":
        if "ivf" not in _retrievers:
            import config
            from pipeline import get_config
            cfg = get_config()
            _retrievers["ivf"] = IVFRetriever(nlist=cfg.ivf_nlist)
        return _retrievers["ivf"]
    elif key == "pq":
        if "pq" not in _retrievers:
            from pipeline import get_config
            cfg = get_config()
            _retrievers["pq"] = PQRetriever(m=cfg.pq_m, nbits=cfg.pq_nbits)
        return _retrievers["pq"]
    else:
        raise ValueError(f"Unknown index type: {key}. Supported: flat, hnsw, ivf, pq")


def build_all_indices(embeddings: np.ndarray) -> None:
    """Build/rebuild all retriever indices with the given embeddings."""
    global _embeddings_store
    _embeddings_store = embeddings.copy()

    # Rebuild all existing retrievers
    for retriever in _retrievers.values():
        retriever.build_index(embeddings)

    # Ensure at least flat is built
    if "flat" not in _retrievers:
        flat = FlatRetriever()
        flat.build_index(embeddings)
        _retrievers["flat"] = flat


def get_stored_embeddings() -> Optional[np.ndarray]:
    """Get the stored document embeddings."""
    return _embeddings_store


def clear_indices() -> None:
    """Clear all indices and stored embeddings."""
    global _embeddings_store
    _retrievers.clear()
    _embeddings_store = None
