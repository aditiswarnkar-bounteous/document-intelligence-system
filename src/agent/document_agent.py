"""Intelligent Document Agent powered by Claude."""

from typing import List, Dict, Optional
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, AGENT_MAX_TOKENS
from .query_planner import QueryPlanner
from .response_synthesizer import ResponseSynthesizer


class DocumentAgent:
    """
    Intelligent agent for document understanding and interaction.
    
    Capabilities:
    - Question answering
    - Information extraction
    - Document comparison
    - Summarization
    - Multi-step reasoning
    """
    
    def __init__(self, retriever, api_key: str = ANTHROPIC_API_KEY):
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        
        self.client = Anthropic(api_key=api_key)
        self.model = CLAUDE_MODEL
        self.retriever = retriever
        self.query_planner = QueryPlanner()
        self.synthesizer = ResponseSynthesizer()
        
        # Conversation history
        self.conversation_history = []
    
    def process_query(
        self,
        query: str,
        mode: str = "qa",
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Process a user query with intelligent routing.
        
        Args:
            query: User query
            mode: Processing mode (qa, extract, summarize, compare)
            context: Additional context
            
        Returns:
            Response dictionary with answer and metadata
        """
        # Plan the query
        query_plan = self.query_planner.plan(query, mode)
        
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query,
            max_results=query_plan['max_chunks']
        )
        
        if not chunks:
            return {
                'answer': "I couldn't find relevant information in the documents to answer your question.",
                'sources': [],
                'confidence': 0.0,
                'mode': mode
            }
        
        # Generate response based on mode
        if mode == "qa":
            response = self._answer_question(query, chunks, query_plan)
        elif mode == "extract":
            response = self._extract_information(query, chunks, query_plan)
        elif mode == "summarize":
            response = self._summarize(query, chunks, query_plan)
        elif mode == "compare":
            response = self._compare(query, chunks, query_plan)
        else:
            response = self._answer_question(query, chunks, query_plan)
        
        # Add to conversation history
        self._add_to_history(query, response)
        
        return response
    
    def _answer_question(
        self,
        query: str,
        chunks: List[Dict],
        plan: Dict
    ) -> Dict:
        """Answer a question using retrieved chunks."""
        context = self.synthesizer.build_context(chunks)
        
        prompt = f"""You are an intelligent document assistant. Answer the user's question based on the provided context from their documents.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, accurate answer based solely on the context
- Cite specific sources when making claims (e.g., "According to [document name]...")
- If information is not in the context, say so clearly
- Be concise but comprehensive
- Use bullet points for multiple items
- Highlight key information

ANSWER:"""
        
        response_text = self._call_claude(prompt)
        
        return {
            'answer': response_text,
            'sources': self._extract_sources(chunks),
            'chunks_used': len(chunks),
            'confidence': self._estimate_confidence(chunks),
            'mode': 'qa'
        }
    
    def _extract_information(
        self,
        query: str,
        chunks: List[Dict],
        plan: Dict
    ) -> Dict:
        """Extract specific information from documents."""
        context = self.synthesizer.build_context(chunks)
        
        prompt = f"""Extract the requested information from the document context.

CONTEXT:
{context}

EXTRACTION REQUEST: {query}

INSTRUCTIONS:
- Extract only factual information present in the documents
- Format as a structured list
- Include document sources for each piece of information
- If information is not found, state this clearly

EXTRACTED INFORMATION:"""
        
        response_text = self._call_claude(prompt)
        
        return {
            'answer': response_text,
            'sources': self._extract_sources(chunks),
            'chunks_used': len(chunks),
            'confidence': self._estimate_confidence(chunks),
            'mode': 'extract'
        }
    
    def _summarize(
        self,
        query: str,
        chunks: List[Dict],
        plan: Dict
    ) -> Dict:
        """Summarize document content."""
        context = self.synthesizer.build_context(chunks)
        
        prompt = f"""Provide a concise summary of the document content.

CONTEXT:
{context}

FOCUS: {query}

INSTRUCTIONS:
- Create a clear, organized summary
- Highlight key points and important details
- Use sections/headers if appropriate
- Keep it concise but informative

SUMMARY:"""
        
        response_text = self._call_claude(prompt)
        
        return {
            'answer': response_text,
            'sources': self._extract_sources(chunks),
            'chunks_used': len(chunks),
            'confidence': self._estimate_confidence(chunks),
            'mode': 'summarize'
        }
    
    def _compare(
        self,
        query: str,
        chunks: List[Dict],
        plan: Dict
    ) -> Dict:
        """Compare information across documents."""
        context = self.synthesizer.build_context(chunks)
        
        prompt = f"""Compare and contrast information from the documents.

CONTEXT:
{context}

COMPARISON REQUEST: {query}

INSTRUCTIONS:
- Identify similarities and differences
- Organize comparison clearly
- Cite specific documents
- Highlight key distinctions

COMPARISON:"""
        
        response_text = self._call_claude(prompt)
        
        return {
            'answer': response_text,
            'sources': self._extract_sources(chunks),
            'chunks_used': len(chunks),
            'confidence': self._estimate_confidence(chunks),
            'mode': 'compare'
        }
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=AGENT_MAX_TOKENS,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks."""
        sources = {}
        for chunk in chunks:
            doc = chunk['document']
            if doc not in sources:
                sources[doc] = {
                    'document': doc,
                    'pages': set(),
                    'relevance': chunk.get('relevance_score', 0)
                }
            sources[doc]['pages'].add(chunk['page_number'])
        
        # Convert to list and format pages
        source_list = []
        for doc_data in sources.values():
            doc_data['pages'] = sorted(list(doc_data['pages']))
            source_list.append(doc_data)
        
        # Sort by relevance
        source_list.sort(key=lambda x: x['relevance'], reverse=True)
        
        return source_list
    
    def _estimate_confidence(self, chunks: List[Dict]) -> float:
        """Estimate confidence in the answer."""
        if not chunks:
            return 0.0
        
        avg_relevance = sum(
            chunk.get('relevance_score', 0) for chunk in chunks
        ) / len(chunks)
        
        # Adjust based on number of chunks
        coverage_factor = min(len(chunks) / 5, 1.0)
        
        confidence = (avg_relevance * 0.7 + coverage_factor * 0.3)
        
        return round(confidence, 2)
    
    def _add_to_history(self, query: str, response: Dict):
        """Add interaction to conversation history."""
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': None  # Could add timestamp
        })
        
        # Keep last 10 interactions
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "Answer questions about documents",
            "Extract specific information",
            "Summarize document content",
            "Compare information across documents",
            "Multi-document reasoning",
            "Source citation and verification"
            ]
        ]
