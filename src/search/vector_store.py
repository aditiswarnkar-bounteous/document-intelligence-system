"""Vector store for semantic document search using TF-IDF."""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path


class VectorStore:
    """In-memory vector store using TF-IDF for semantic search."""
    
    def __init__(self):
        self.chunks: List[Dict] = []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectors = None
        self.is_indexed = False
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of document chunks
        """
        self.chunks = chunks
        self._build_index()
    
    def _build_index(self):
        """Build the vector index."""
        if not self.chunks:
            return
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in self.chunks]
        
        # Build TF-IDF vectors
        self.vectors = self.vectorizer.fit_transform(texts)
        self.is_indexed = True
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.is_indexed:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                chunk = self.chunks[idx].copy()
                results.append((chunk, float(similarity)))
        
        return results
    
    def save(self, path: Path):
        """Save the vector store to disk."""
        data = {
            'chunks': self.chunks,
            'vectorizer': self.vectorizer,
            'vectors': self.vectors
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: Path):
        """Load the vector store from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.vectorizer = data['vectorizer']
        self.vectors = data['vectors']
        self.is_indexed = True
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'total_chunks': len(self.chunks),
            'is_indexed': self.is_indexed,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.is_indexed else 0,
            'documents': len(set(chunk['document'] for chunk in self.chunks))
        }
        }
