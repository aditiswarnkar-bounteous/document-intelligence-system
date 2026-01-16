"""Document Intelligence Agent - Streamlit Application."""

import streamlit as st
from pathlib import Path
import time

from config import (
    DOCUMENTS_DIR, PAGE_TITLE, PAGE_ICON, LAYOUT
)
from src.processors.pdf_processor import load_all_documents
from src.processors.text_chunker import chunk_documents
from src.search.vector_store import VectorStore
from src.search.retriever import Retriever
from src.agent.document_agent import DocumentAgent


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_system():
    """Initialize the document intelligence system."""
    with st.spinner("🔄 Initializing Document Intelligence System..."):
        # Load documents
        st.write("📄 Loading documents...")
        documents = load_all_documents(DOCUMENTS_DIR)
        
        if not documents:
            return None, None, []
        
        # Chunk documents
        st.write("✂️ Creating intelligent chunks...")
        chunks = chunk_documents(documents)
        
        # Build vector store
        st.write("🔍 Building search index...")
        vector_store = VectorStore()
        vector_store.add_chunks(chunks)
        
        # Create retriever
        retriever = Retriever(vector_store)
        
        # Create agent
        st.write("🤖 Initializing AI agent...")
        agent = DocumentAgent(retriever)
        
        return agent, vector_store, documents


def render_header():
    """Render application header."""
    st.title("🤖 Document Intelligence Agent")
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
        font-weight: 500;
    }
    </style>
    <p class="big-font">
    Your AI-powered assistant for intelligent document analysis and question answering.
    </p>
    """, unsafe_allow_html=True)


def render_sidebar(agent, vector_store, documents):
    """Render sidebar with system info and controls."""
    with st.sidebar:
        st.header("📚 Document Library")
        
        # Document stats
        if documents:
            st.metric("Documents Loaded", len(documents))
            
            for doc in documents:
                with st.expander(f"📄 {doc['file_name']}", expanded=False):
                    st.write(f"**Pages:** {doc['total_pages']}")
                    if doc['metadata'].get('title'):
                        st.write(f"**Title:** {doc['metadata']['title']}")
        
        st.divider()
        
        # System stats
        if vector_store:
            st.header("📊 System Status")
            stats = vector_store.get_stats()
            col1, col2 = st.columns(2)
            col1.metric("Chunks", stats['total_chunks'])
            col2.metric("Vocabulary", stats['vocabulary_size'])
        
        st.divider()
        
        # Agent capabilities
        if agent:
            st.header("🎯 Capabilities")
            for capability in agent.get_capabilities():
                st.write(f"✅ {capability}")
        
        st.divider()
        
        # Mode selector
        st.header("⚙️ Query Mode")
        mode = st.selectbox(
            "Select mode:",
            ["qa", "extract", "summarize", "compare"],
            format_func=lambda x: {
                "qa": "❓ Question Answering",
                "extract": "📋 Information Extraction",
                "summarize": "📝 Summarization",
                "compare": "⚖️ Comparison"
            }[x]
        )
        
        return mode


def render_query_interface(agent, mode):
    """Render main query interface."""
    
    # Query input
    st.subheader("💬 Ask Your Question")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Query:",
            placeholder="e.g., What is the registered office address of Bandhan Bank?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process query
    if search_clicked and query:
        process_query(agent, query, mode)
    
    # Example queries
    render_examples()


def process_query(agent, query, mode):
    """Process and display query results."""
    
    # Create placeholder for status
    status_placeholder = st.empty()
    
    with status_placeholder.container():
        with st.spinner("🤔 Processing your query..."):
            start_time = time.time()
            
            # Get response from agent
            response = agent.process_query(query, mode=mode)
            
            elapsed_time = time.time() - start_time
    
    # Clear status
    status_placeholder.empty()
    
    # Display results
    st.success(f"✅ Response generated in {elapsed_time:.2f}s")
    
    # Main answer
    st.markdown("### 💡 Answer")
    st.markdown(response['answer'])
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{response['confidence']:.0%}")
    col2.metric("Sources Used", len(response['sources']))
    col3.metric("Chunks Analyzed", response['chunks_used'])
    
    st.divider()
    
    # Sources
    if response['sources']:
        st.markdown("### 📚 Sources")
        
        for source in response['sources']:
            with st.expander(
                f"📄 {source['document']} - Pages: {', '.join(map(str, source['pages']))}",
                expanded=False
            ):
                st.write(f"**Relevance Score:** {source['relevance']:.2%}")
                
                # Show sample chunk
                st.caption("Sample excerpt:")
                # Could add chunk preview here if needed


def render_examples():
    """Render example queries."""
    st.divider()
    st.markdown("### 💡 Example Questions")
    
    examples = [
        "What is the registered office address of Bandhan Bank?",
        "What documents are required for KYC verification?",
        "What are the powers and duties of directors?",
        "List all the committees mentioned in the AOA",
        "What is the share capital structure?",
        "Summarize the key provisions about dividends"
    ]
    
    cols = st.columns(3)
    for idx, example in enumerate(examples):
        with cols[idx % 3]:
            if st.button(example, key=f"ex_{idx}", use_container_width=True):
                st.session_state.query = example
                st.rerun()


def main():
    """Main application function."""
    
    # Initialize system
    agent, vector_store, documents = initialize_system()
    
    # Check if system initialized
    if agent is None:
        st.error("⚠️ No documents found!")
        st.info(f"Please add PDF documents to: `{DOCUMENTS_DIR}`")
        st.stop()
    
    # Render UI
    render_header()
    
    mode = render_sidebar(agent, vector_store, documents)
    
    render_query_interface(agent, mode)
    
    # Footer
    st.divider()
    st.caption("Powered by Claude Sonnet 4 🚀 | Built with Streamlit")


if __name__ == "__main__":
    main()
