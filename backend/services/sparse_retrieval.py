"""Sparse retrieval service using TF-IDF."""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SparseRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tf_idf_matrix = None
        self.is_built = False

    def build_index(self, texts: list[str]):
        if not texts:
            self.is_built = False
            return
        self.tf_idf_matrix = self.vectorizer.fit_transform(texts)
        self.is_built = True

    def search(self, query: str, top_k: int) -> tuple[list[float], list[int]]:
        """Return (scores, indices) of top_k results."""
        if not self.is_built:
            return [], []
            
        query_vec = self.vectorizer.transform([query])
        similarities = (self.tf_idf_matrix * query_vec.T).toarray().flatten()
        
        # Argsort ascending, then reverse to descending
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        
        # Filter out absolute 0s
        valid_mask = top_scores > 0.0
        return top_scores[valid_mask].tolist(), top_indices[valid_mask].tolist()

sparse_retriever = SparseRetriever()
