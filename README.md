# Document-intelligence-system
AI-powered document understanding system
An advanced AI-powered document intelligence system that enables sophisticated document understanding, analysis, and interaction.

###  Features

### Core Capabilities
- ** Intelligent Search**: Semantic search across multiple documents using TF-IDF vectorization
- ** Question Answering**: Natural language Q&A with source citations
- ** Information Extraction**: Extract specific data points from documents
- ** Summarization**: Generate concise summaries of document content
- ** Comparison**: Compare information across multiple documents
- ** Multi-step Reasoning**: Complex query understanding and planning

### Technical Features
- Smart document chunking with context preservation
- Vector-based semantic search
- Query planning and optimization
- Confidence scoring
- Source citation tracking
- Conversation history
- Beautiful Streamlit UI

##  Architecture

```
┌─────────────────┐
│   Streamlit UI  │
└────────┬────────┘
         │
┌────────▼────────────┐
│  Document Agent     │ ← Main Intelligence Layer
│  - Query Planner    │
│  - Claude API       │
│  - Response Synth   │
└────────┬────────────┘
         │
┌────────▼────────────┐
│    Retriever        │ ← Search & Ranking
│  - Query Enhancement│
│  - Result Reranking │
└────────┬────────────┘
         │
┌────────▼────────────┐
│   Vector Store      │ ← TF-IDF Index
│  - Similarity Search│
└────────┬────────────┘
         │
┌────────▼────────────┐
│   Text Chunker      │ ← Document Processing
│  - Smart Splitting  │
└────────┬────────────┘
         │
┌────────▼────────────┐
│  PDF Processor      │ ← Document Loading
└─────────────────────┘
```

### Prerequisites
- Python 3.8 or higher
- Anthropic API key 

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/document-intelligence-agent.git
cd document-intelligence-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment**
```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

5. **Add your documents**
```bash
# Place your PDF files in data/documents/
cp your_documents/*.pdf data/documents/
```

6. **Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

##  Usage Guide

### Basic Question Answering
```
Question: "What is the registered office address?"
Mode: Question Answering

The agent will:
1. Understand your query
2. Search relevant document sections
3. Generate accurate answer with citations
```

### Information Extraction
```
Query: "Extract all director names and their roles"
Mode: Information Extraction

Returns structured list of information with sources
```

### Document Summarization
```
Query: "Summarize the key points about share capital"
Mode: Summarization

Provides concise summary of relevant sections
```

### Comparison
```
Query: "Compare the requirements in both documents"
Mode: Comparison

Analyzes and contrasts information across documents
```

##  Query Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **QA** | Question answering | Specific questions |
| **Extract** | Information extraction | Finding data points |
| **Summarize** | Content summarization | Overview and key points |
| **Compare** | Cross-document comparison | Finding differences |

##  Configuration

Edit `config.py` to customize:

```python
# Document Processing
CHUNK_SIZE = 2000           # Characters per chunk
CHUNK_OVERLAP = 300         # Overlap between chunks

# Search
MAX_SEARCH_RESULTS = 8      # Results to retrieve
RELEVANCE_THRESHOLD = 0.3   # Minimum relevance score

# Agent
AGENT_TEMPERATURE = 0.7     # Response creativity
AGENT_MAX_TOKENS = 3000     # Max response length
```

##  Project Structure

```
document-intelligence-agent/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── documents/             # Your PDF files go here
│   └── index/                 # Generated search indices
│
└── src/
    ├── agent/                 # AI agent components
    │   ├── document_agent.py  # Main agent logic
    │   ├── query_planner.py   # Query analysis
    │   └── response_synthesizer.py
    │
    ├── processors/            # Document processing
    │   ├── pdf_processor.py   # PDF loading
    │   └── text_chunker.py    # Smart chunking
    │
    ├── search/                # Search system
    │   ├── vector_store.py    # TF-IDF index
    │   └── retriever.py       # Search & ranking
    │
    └── utils/                 # Utilities
        └── helpers.py
```

##  Features in Detail

### Smart Chunking
- Respects paragraph boundaries
- Maintains context with overlap
- Preserves document structure
- Metadata tracking (page numbers, sources)

### Intelligent Search
- TF-IDF semantic similarity
- Query enhancement with synonyms
- Result reranking with multiple factors
- Relevance scoring and thresholding

### AI Agent Capabilities
- Query intent understanding
- Multi-step reasoning
- Context synthesis from multiple sources
- Confidence estimation
- Source citation

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  Example Queries

**Question Answering:**
- "What is the registered office address of Bandhan Bank?"
- "Who are the directors of the company?"
- "What is the share capital structure?"

**Information Extraction:**
- "Extract all committee names and their purposes"
- "List all the documents required for KYC"
- "Find all references to dividend policies"

**Summarization:**
- "Summarize the articles of association"
- "Give an overview of the KYC requirements"
- "Summarize director responsibilities"

**Comparison:**
- "Compare KYC requirements across documents"
- "What are the differences in board structures?"

##  Troubleshooting

**No documents found:**
- Ensure PDF files are in `data/documents/`
- Check file permissions

**API errors:**
- Verify ANTHROPIC_API_KEY in `.env`
- Check API key validity
- Ensure internet connection

**Poor search results:**
- Adjust RELEVANCE_THRESHOLD in config.py
- Increase MAX_SEARCH_RESULTS
- Try rephrasing your query



---

**Made with ❤️ and AI**
