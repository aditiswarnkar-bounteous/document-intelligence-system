"""Advanced retrieval with query enhancement and reranking."""

from typing import List, Dict, Tuple
from .vector_store import VectorStore
from config import MAX_SEARCH_RESULTS, RELEVANCE_THRESHOLD


class Retriever:
    """Advanced retriever with query processing and result ranking."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        max_results: int = MAX_SEARCH_RESULTS,
        threshold: float = RELEVANCE_THRESHOLD,
        filter_document: str = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            max_results: Maximum number of results
            threshold: Minimum relevance threshold
            filter_document: Optional document name to filter by
            
        Returns:
            List of relevant chunks with scores
        """
        # Enhance query
        enhanced_query = self._enhance_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            enhanced_query,
            top_k=max_results * 2,  # Get more, then filter
            threshold=threshold
        )
        
        # Filter by document if specified
        if filter_document:
            results = [
                (chunk, score) for chunk, score in results
                if chunk['document'] == filter_document
            ]
        
        # Rerank results
        reranked = self._rerank_results(query, results)
        
        # Format results
        formatted_results = []
        for chunk, score in reranked[:max_results]:
            chunk['relevance_score'] = score
            formatted_results.append(chunk)
        
        return formatted_results
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance query with synonyms and expansions.
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query
        """
        # Add common banking/document terms if relevant
        enhancements = {
            'address': 'address registered office location',
            'director': 'director board member officer',
            'kyc': 'kyc know your customer verification',
            'document': 'document form certificate',
        }
        
        query_lower = query.lower()
        enhanced = query
        
        for key, expansion in enhancements.items():
            if key in query_lower:
                enhanced = f"{query} {expansion}"
                break
        
        return enhanced
    
    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[Dict, float]]
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank results based on additional factors.
        
        Args:
            query: Original query
            results: Initial search results
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        query_terms = set(query.lower().split())
        
        scored_results = []
        for chunk, base_score in results:
            # Calculate additional scoring factors
            text_lower = chunk['text'].lower()
            
            # Exact phrase match bonus
            phrase_bonus = 0.2 if query.lower() in text_lower else 0
            
            # Term frequency bonus
            term_matches = sum(1 for term in query_terms if term in text_lower)
            term_bonus = (term_matches / len(query_terms)) * 0.1
            
            # Length penalty (prefer more substantial chunks)
            length_factor = min(chunk['char_count'] / 1000, 1.0) * 0.05
            
            # Calculate final score
            final_score = base_score + phrase_bonus + term_bonus + length_factor
            
            scored_results.append((chunk, final_score))
        
        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return scored_results
    
    def get_document_chunks(self, document_name: str) -> List[Dict]:
        """
        Get all chunks from a specific document.
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of chunks from that document
        """
        return [
            chunk for chunk in self.vector_store.chunks
            if chunk['document'] == document_name
            ]
        ]
