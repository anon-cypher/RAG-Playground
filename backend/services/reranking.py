"""Re-ranking module using Reciprocal Rank Fusion (RRF)."""

def reciprocal_rank_fusion(dense_indices, dense_scores, sparse_indices, sparse_scores, k=60, top_k=None):
    """
    Combines dense FAISS ranks with sparse TF-IDF ranks mathematically.
    Higher dense_score = worse if it's L2 distance. So rank based on ascending L2 distance.
    Higher sparse_score = better if it's Cosine Sim. So rank based on descending Cosine sim.
    This function expects dense_indices to already be ordered by best dense rank, 
    and sparse_indices to already be ordered by best sparse rank.
    """
    rrf_scores = {}
    
    # Dense ranks
    for rank, idx in enumerate(dense_indices):
        if idx not in rrf_scores:
            rrf_scores[idx] = 0.0
        rrf_scores[idx] += 1.0 / (k + rank + 1)
        
    # Sparse ranks
    for rank, idx in enumerate(sparse_indices):
        if idx not in rrf_scores:
            rrf_scores[idx] = 0.0
        rrf_scores[idx] += 1.0 / (k + rank + 1)
        
    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    if top_k:
        sorted_items = sorted_items[:top_k]
        
    final_indices = [item[0] for item in sorted_items]
    final_scores = [item[1] for item in sorted_items]  # RRF scores
    
    return final_indices, final_scores
