"""Configuration for Document Intelligence System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# API Configuration
# ============================================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"

# Create directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Document Processing
# ============================================================================
CHUNK_SIZE = 2000          # Characters per chunk
CHUNK_OVERLAP = 300        # Overlap between chunks
MIN_CHUNK_SIZE = 100       # Minimum chunk size to keep

# ============================================================================
# Search & Retrieval
# ============================================================================
MAX_SEARCH_RESULTS = 8     # Maximum chunks to retrieve
RELEVANCE_THRESHOLD = 0.3  # Minimum relevance score

# ============================================================================
# Agent Configuration
# ============================================================================
AGENT_TEMPERATURE = 0.7
AGENT_MAX_TOKENS = 3000
AGENT_TIMEOUT = 60         # Seconds

# Agent capabilities
AGENT_CAPABILITIES = [
    "document_search",
    "information_extraction",
    "summarization",
    "comparison",
    "question_answering",
]

# ============================================================================
# Streamlit UI
# ============================================================================
PAGE_TITLE = "Document Intelligence Agent"
PAGE_ICON = "🤖"
LAYOUT = "wide"

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
