"""Intelligent text chunking with semantic awareness."""

from typing import List, Dict
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


class TextChunker:
    """Smart text chunking that respects document structure."""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_document(self, document_data: Dict) -> List[Dict]:
        """
        Chunk a document intelligently.
        
        Args:
            document_data: Document data from PDFProcessor
            
        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        chunk_id = 0
        
        for page_data in document_data['pages']:
            page_chunks = self._chunk_page(
                text=page_data['text'],
                page_number=page_data['page_number'],
                source=document_data['file_name']
            )
            
            for chunk in page_chunks:
                chunk['chunk_id'] = chunk_id
                chunk['document'] = document_data['file_name']
                all_chunks.append(chunk)
                chunk_id += 1
        
        return all_chunks
    
    def _chunk_page(self, text: str, page_number: int, source: str) -> List[Dict]:
        """Chunk a single page of text."""
        chunks = []
        
        # Try to split by paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk,
                        page_number,
                        source,
                        current_start
                    ))
                
                # Start new chunk with overlap
                if len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + para
                else:
                    current_chunk = para
                
                current_start += len(current_chunk) - self.chunk_overlap
            else:
                current_chunk += para
        
        # Add remaining chunk
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                current_chunk,
                page_number,
                source,
                current_start
            ))
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        return [p.strip() + '\n\n' for p in paragraphs if p.strip()]
    
    def _create_chunk(
        self,
        text: str,
        page_number: int,
        source: str,
        start_pos: int
    ) -> Dict:
        """Create a chunk with metadata."""
        return {
            'text': text.strip(),
            'page_number': page_number,
            'source': source,
            'start_position': start_pos,
            'char_count': len(text.strip()),
            'word_count': len(text.strip().split())
        }


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Chunk all documents.
    
    Args:
        documents: List of document data
        
    Returns:
        List of all chunks from all documents
    """
    chunker = TextChunker()
    all_chunks = []
    
    for doc in documents:
        doc_chunks = chunker.chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"  → Created {len(doc_chunks)} chunks from {doc['file_name']}")
    
    return all_chunks
    
    return all_chunks
