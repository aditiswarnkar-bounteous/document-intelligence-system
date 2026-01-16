"""Response synthesis and formatting."""

from typing import List, Dict


class ResponseSynthesizer:
    """Synthesize responses from multiple document chunks."""
    
    def build_context(
        self,
        chunks: List[Dict],
        max_chars: int = 8000
    ) -> str:
        """
        Build context string from chunks.
        
        Args:
            chunks: List of document chunks
            max_chars: Maximum context characters
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_chars = 0
        
        # Group chunks by document
        chunks_by_doc = self._group_by_document(chunks)
        
        for doc_name, doc_chunks in chunks_by_doc.items():
            doc_context = self._format_document_context(doc_name, doc_chunks)
            
            if total_chars + len(doc_context) > max_chars:
                break
            
            context_parts.append(doc_context)
            total_chars += len(doc_context)
        
        return "\n\n".join(context_parts)
    
    def _group_by_document(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group chunks by source document."""
        grouped = {}
        
        for chunk in chunks:
            doc = chunk['document']
            if doc not in grouped:
                grouped[doc] = []
            grouped[doc].append(chunk)
        
        return grouped
    
    def _format_document_context(
        self,
        document_name: str,
        chunks: List[Dict]
    ) -> str:
        """Format context for a single document."""
        parts = [f"=== {document_name} ===\n"]
        
        for chunk in chunks:
            page_info = f"[Page {chunk['page_number']}]"
            relevance = chunk.get('relevance_score', 0)
            relevance_indicator = self._get_relevance_indicator(relevance)
            
            parts.append(
                f"{page_info} {relevance_indicator}\n{chunk['text']}\n"
            )
        
        return "\n".join(parts)
    
    def _get_relevance_indicator(self, score: float) -> str:
        """Get visual indicator of relevance."""
        if score >= 0.7:
            return "⭐⭐⭐"
        elif score >= 0.5:
            return "⭐⭐"
        elif score >= 0.3:
            return "⭐"
        return ""
    
    def format_sources(self, sources: List[Dict]) -> str:
        """Format source citations."""
        if not sources:
            return "No sources"
        
        formatted = []
        for source in sources:
            doc = source['document']
            pages = source['pages']
            
            if len(pages) == 1:
                page_str = f"page {pages[0]}"
            else:
                page_str = f"pages {', '.join(map(str, pages))}"
            
            formatted.append(f"• {doc} ({page_str})")
        
        return "\n".join(formatted)
    
    def create_summary_response(self, chunks: List[Dict]) -> str:
        """Create a brief summary of what was found."""
        if not chunks:
            return "No relevant information found."
        
        docs = set(chunk['document'] for chunk in chunks)
        total_pages = len(set(
            (chunk['document'], chunk['page_number']) for chunk in chunks
        ))
        
        return f"Found {len(chunks)} relevant passages from {len(docs)} document(s) across {total_pages} pages."
