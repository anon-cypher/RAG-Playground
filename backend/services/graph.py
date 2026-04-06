"""Knowledge Graph extraction and retrieval using NetworkX."""
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.feature_names = []
        self.is_built = False

    def build_graph(self, chunks: list):
        """Builds a bipartite graph of Chunks and Keywords, and explicit Chunk-Chunk edges."""
        self.graph.clear()
        if not chunks:
            self.is_built = False
            return
            
        texts = [c.text for c in chunks]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Add chunk nodes
        for c in chunks:
            self.graph.add_node(f"chunk_{c.chunk_id}", type="chunk", doc_id=c.document_id)
            
        # Add edges based on top keywords
        for i, row in enumerate(tfidf_matrix.toarray()):
            chunk_node = f"chunk_{chunks[i].chunk_id}"
            # Get top 5 keywords for this chunk
            top_kw_indices = np.argsort(row)[::-1][:5]
            for kw_idx in top_kw_indices:
                score = row[kw_idx]
                if score > 0.1:
                    keyword = self.feature_names[kw_idx]
                    kw_node = f"kw_{keyword}"
                    if not self.graph.has_node(kw_node):
                        self.graph.add_node(kw_node, type="keyword", word=keyword)
                    self.graph.add_edge(chunk_node, kw_node, weight=score)
                    
        self.is_built = True

    def retrieve_subgraph(self, query: str, top_k: int) -> list[str]:
        """Extract main keywords from query and return connected chunk IDs."""
        if not self.is_built:
            return []
            
        try:
            query_vec = self.vectorizer.transform([query]).toarray()[0]
        except Exception:
            return []
            
        top_kw_indices = np.argsort(query_vec)[::-1][:3]
        query_keywords = [self.feature_names[i] for i in top_kw_indices if query_vec[i] > 0]
        
        chunk_scores = {}
        # Traverse graph 1 hop from keywords to find relevant chunks
        for kw in query_keywords:
            kw_node = f"kw_{kw}"
            if self.graph.has_node(kw_node):
                for neighbor in self.graph.neighbors(kw_node):
                    if self.graph.nodes[neighbor].get("type") == "chunk":
                        weight = self.graph[kw_node][neighbor].get("weight", 0.1)
                        chunk_id = neighbor.replace("chunk_", "")
                        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight
                        
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_chunks[:top_k]]

knowledge_graph = KnowledgeGraph()
