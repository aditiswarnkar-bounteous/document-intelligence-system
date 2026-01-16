"""Advanced PDF document processor with metadata extraction."""

from pathlib import Path
from typing import Dict, List, Optional
import PyPDF2
from datetime import datetime


class PDFProcessor:
    """Process PDF documents and extract text with metadata."""
    
    def __init__(self):
        self.supported_extensions = ['.pdf']
    
    def load_document(self, file_path: Path) -> Dict:
        """
        Load a PDF document with metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary containing text, metadata, and structure
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        return self._process_pdf(file_path)
    
    def _process_pdf(self, file_path: Path) -> Dict:
        """Process a PDF file and extract all information."""
        document_data = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'pages': [],
            'metadata': {},
            'total_pages': 0,
            'processed_at': datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                document_data['metadata'] = self._extract_metadata(pdf_reader)
                document_data['total_pages'] = len(pdf_reader.pages)
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        document_data['pages'].append({
                            'page_number': page_num,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
        
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")
        
        return document_data
    
    def _extract_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict:
        """Extract metadata from PDF."""
        metadata = {}
        
        if pdf_reader.metadata:
            metadata_fields = [
                'author', 'creator', 'producer', 'subject',
                'title', 'creation_date', 'modification_date'
            ]
            
            for field in metadata_fields:
                value = getattr(pdf_reader.metadata, field, None)
                if value:
                    metadata[field] = str(value)
        
        return metadata
    
    def get_document_summary(self, document_data: Dict) -> str:
        """Generate a summary of the document."""
        summary_parts = [
            f"Document: {document_data['file_name']}",
            f"Pages: {document_data['total_pages']}",
        ]
        
        if document_data['metadata'].get('title'):
            summary_parts.insert(1, f"Title: {document_data['metadata']['title']}")
        
        total_chars = sum(page['char_count'] for page in document_data['pages'])
        summary_parts.append(f"Total characters: {total_chars:,}")
        
        return "\n".join(summary_parts)


def load_all_documents(documents_dir: Path) -> List[Dict]:
    """
    Load all PDF documents from a directory.
    
    Args:
        documents_dir: Directory containing PDF files
        
    Returns:
        List of processed document data
    """
    processor = PDFProcessor()
    documents = []
    
    if not documents_dir.exists():
        return documents
    
    pdf_files = list(documents_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            doc_data = processor.load_document(pdf_file)
            documents.append(doc_data)
            print(f"✓ Loaded: {pdf_file.name}")
        except Exception as e:
            print(f"✗ Failed to load {pdf_file.name}: {e}")
    
    return documents
