"""Query planning and analysis."""

from typing import Dict
import re


class QueryPlanner:
    """Analyze and plan query execution strategy."""
    
    def __init__(self):
        self.question_patterns = {
            'what': ['what', 'which'],
            'who': ['who'],
            'where': ['where', 'location'],
            'when': ['when', 'date', 'time'],
            'how': ['how'],
            'why': ['why', 'reason'],
            'list': ['list', 'enumerate', 'all'],
            'compare': ['compare', 'difference', 'versus', 'vs'],
            'summarize': ['summarize', 'summary', 'overview']
        }
    
    def plan(self, query: str, mode: str = "qa") -> Dict:
        """
        Create an execution plan for a query.
        
        Args:
            query: User query
            mode: Processing mode
            
        Returns:
            Execution plan dictionary
        """
        query_lower = query.lower()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Determine complexity
        complexity = self._assess_complexity(query)
        
        # Decide number of chunks needed
        max_chunks = self._determine_chunk_count(query_type, complexity)
        
        # Extract entities/keywords
        keywords = self._extract_keywords(query)
        
        plan = {
            'query_type': query_type,
            'complexity': complexity,
            'max_chunks': max_chunks,
            'keywords': keywords,
            'mode': mode,
            'requires_multi_doc': self._requires_multi_document(query_lower),
            'requires_synthesis': complexity == 'high'
        }
        
        return plan
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of question."""
        for q_type, patterns in self.question_patterns.items():
            if any(pattern in query for pattern in patterns):
                return q_type
        return 'general'
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        indicators = {
            'high': ['compare', 'analyze', 'explain', 'relationship', 'how does'],
            'medium': ['list', 'describe', 'what are', 'tell me about'],
            'low': ['what is', 'where', 'when', 'who']
        }
        
        query_lower = query.lower()
        
        for level, patterns in indicators.items():
            if any(pattern in query_lower for pattern in patterns):
                return level
        
        # Default based on length
        return 'high' if len(query.split()) > 10 else 'medium'
    
    def _determine_chunk_count(self, query_type: str, complexity: str) -> int:
        """Determine optimal number of chunks to retrieve."""
        base_counts = {
            'low': 3,
            'medium': 5,
            'high': 8
        }
        
        count = base_counts.get(complexity, 5)
        
        # Adjust for query type
        if query_type in ['list', 'compare', 'summarize']:
            count += 2
        
        return min(count, 10)  # Cap at 10
    
    def _extract_keywords(self, query: str) -> list:
        """Extract important keywords from query."""
        # Remove common stop words
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'about', 'tell', 'me'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Limit to top 10
    
    def _requires_multi_document(self, query: str) -> bool:
        """Check if query requires multiple documents."""
        multi_doc_indicators = [
            'compare', 'both', 'all documents', 'across',
            'different', 'versus', 'vs'
          ]
        
        return any(indicator in query for indicator in multi_doc_indicators)
        
        return any(indicator in query for indicator in multi_doc_indicators)
