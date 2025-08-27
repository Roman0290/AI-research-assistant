import streamlit as st
import os
import logging
from typing import List, Dict, Any
import tempfile

# Import core modules
from core.ingestion import (
    load_arxiv_paper, load_wikipedia_page, load_web_url, 
    chunk_documents, validate_source_input
)
from core.embeddings import get_embedding_model
from core.retriever import build_faiss_index, get_retriever, get_index_stats
from core.generator import (
    PromptTemplates, get_llm, build_rag_chain, 
    get_available_models as get_available_llm_models
)
from core.reranker import get_default_reranker
from core.document_manager import DocumentManager
from utils.helpers import ConfigManager, validate_environment, format_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "document_manager" not in st.session_state:
        st.session_state.document_manager = DocumentManager()
    
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "qa"
    
    if "reranker_enabled" not in st.session_state:
        st.session_state.reranker_enabled = False
    
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 500
    
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 50

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🤖 AI Research Assistant (RAG)</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Check environment variables
    env_status = validate_environment()
    
    # Sidebar for configuration and document management
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment status
        st.subheader("Environment Status")
        if env_status.get("GROQ_API_KEY", False):
            st.markdown('<p class="status-success">✅ GROQ API Key: Configured</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ GROQ API Key: Not Set</p>', unsafe_allow_html=True)
            st.info("Please set GROQ_API_KEY environment variable")
        
        # Model configuration
        st.subheader("Model Configuration")
        llm_model = st.selectbox(
            "LLM Model",
            get_available_llm_models(),
            index=0
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        # Chunking configuration
        st.subheader("Document Processing")
        st.session_state.chunk_size = st.number_input(
            "Chunk Size", 
            min_value=100, 
            max_value=2000, 
            value=st.session_state.chunk_size,
            step=50
        )
        
        st.session_state.chunk_overlap = st.number_input(
            "Chunk Overlap", 
            min_value=0, 
            max_value=500, 
            value=st.session_state.chunk_overlap,
            step=10
        )
        
        # Reranker configuration
        st.subheader("Advanced Features")
        st.session_state.reranker_enabled = st.checkbox("Enable Reranker", value=st.session_state.reranker_enabled)
        
        # Document set management
        st.header("📚 Document Sets")
        
        # Create new document set
        with st.expander("Create New Document Set"):
            new_set_name = st.text_input("Set Name", key="new_set_name")
            new_set_description = st.text_area("Description", key="new_set_desc")
            new_set_source = st.selectbox("Source Type", ["Arxiv", "Wikipedia", "Web URL", "Mixed"], key="new_set_source")
            
            if st.button("Create Document Set") and new_set_name:
                try:
                    st.session_state.document_manager.create_document_set(
                        new_set_name, new_set_description, new_set_source
                    )
                    st.success(f"Created document set: {new_set_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating document set: {e}")
        
        # Document set selection
        available_sets = st.session_state.document_manager.list_document_sets()
        if available_sets:
            set_names = [set_info["name"] for set_info in available_sets]
            current_set = st.selectbox(
                "Select Document Set",
                set_names,
                index=set_names.index(st.session_state.document_manager.active_set) if st.session_state.document_manager.active_set else 0
            )
            
            if st.button("Switch to Selected Set"):
                st.session_state.document_manager.switch_document_set(current_set)
                st.success(f"Switched to document set: {current_set}")
                st.rerun()
            
            # Show current set info
            current_set_info = st.session_state.document_manager.get_document_set_info(current_set)
            if current_set_info:
                st.info(f"Active: {current_set_info['name']} ({current_set_info['document_count']} documents)")
        else:
            st.info("No document sets available. Create one to get started.")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Document Ingestion", "🔍 Q&A Mode", "📝 Summary Mode", "📊 Analysis Mode"])
    
    # Tab 1: Document Ingestion
    with tab1:
        st.header("📖 Document Ingestion")
        
        if not env_status.get("GROQ_API_KEY", False):
            st.error("Please configure your GROQ API key to continue.")
            return
        
        # Document source selection
        col1, col2 = st.columns(2)
        
        with col1:
            source = st.selectbox("Choose source", ["Arxiv", "Wikipedia", "Web URL"])
            query = st.text_input("Enter query (Arxiv ID, Wikipedia title, or URL)")
            
            # Source-specific help
            if source == "Arxiv":
                st.info("Enter Arxiv paper ID (e.g., 2303.08774) or search query")
            elif source == "Wikipedia":
                st.info("Enter Wikipedia page title or search term")
            else:
                st.info("Enter a valid URL starting with http:// or https://")
        
        with col2:
            # Document set selection for ingestion
            if available_sets:
                target_set = st.selectbox(
                    "Target Document Set",
                    [set_info["name"] for set_info in available_sets]
                )
            else:
                st.warning("No document sets available. Please create one first.")
                target_set = None
        
        # Load and process documents
        if st.button("Load & Process Documents") and query and target_set:
            if not validate_source_input(source, query):
                st.error("Invalid input. Please check your query format.")
                return
            
            try:
                with st.spinner("Loading and processing documents..."):
                    # Load documents based on source
                    if source == "Arxiv":
                        docs = load_arxiv_paper(query)
                    elif source == "Wikipedia":
                        docs = load_wikipedia_page(query)
                    else:
                        docs = load_web_url(query)
                    
                    if not docs:
                        st.error("No documents found. Please check your query.")
                        return
                    
                    # Chunk documents
                    chunks = chunk_documents(
                        docs, 
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap
                    )
                    
                    # Get embeddings model
                    embeddings = get_embedding_model()
                    
                    # Add to document set
                    success = st.session_state.document_manager.add_documents_to_set(
                        target_set, chunks, embeddings
                    )
                    
                    if success:
                        st.success(f"Successfully processed {len(chunks)} chunks and added to '{target_set}'")
                        
                        # Show document set info
                        set_info = st.session_state.document_manager.get_document_set_info(target_set)
                        if set_info:
                            st.json(set_info)
                    else:
                        st.error("Failed to add documents to document set")
                        
            except Exception as e:
                st.error(f"Error processing documents: {e}")
                logger.error(f"Document processing error: {e}")
    
    # Tab 2: Q&A Mode
    with tab2:
        st.header("🔍 Q&A Mode")
        st.session_state.current_mode = "qa"
        
        if not st.session_state.document_manager.get_active_set():
            st.warning("Please load and index documents first.")
        else:
            # Show active document set info
            active_set = st.session_state.document_manager.get_active_set()
            st.info(f"Active Document Set: {active_set.name} ({active_set.document_count} documents)")
            
            # Q&A interface
            user_question = st.text_input("Ask a question about your documents:")
            
            if st.button("Get Answer") and user_question:
                try:
                    with st.spinner("Generating answer..."):
                        # Get active FAISS index
                        faiss_index = st.session_state.document_manager.get_active_faiss_index()
                        if not faiss_index:
                            st.error("No FAISS index available for the active document set.")
                            return
                        
                        # Get retriever
                        retriever = get_retriever(faiss_index)
                        
                        # Apply reranking if enabled
                        if st.session_state.reranker_enabled:
                            embeddings = get_embedding_model()
                            reranker = get_default_reranker(embeddings)
                            
                            # Get initial retrieval
                            initial_docs = retriever.retrieve(user_question)
                            
                            # Apply reranking
                            reranked_docs = reranker.rerank(user_question, initial_docs)
                            
                            # Update retriever with reranked results
                            class RerankedRetriever:
                                def __init__(self, docs):
                                    self.docs = docs
                                
                                def retrieve(self, query):
                                    return self.docs
                            
                            retriever = RerankedRetriever(reranked_docs)
                        
                        # Get LLM and build chain
                        llm = get_llm(llm_model, temperature)
                        prompt = PromptTemplates.get_qa_template()
                        rag_chain = build_rag_chain(retriever, llm, prompt, "qa")
                        
                        # Generate answer
                        answer = rag_chain.invoke({"question": user_question})
                        
                        # Display answer
                        st.markdown("### Answer")
                        st.write(answer)
                        
                        # Show retrieved context
                        with st.expander("View Retrieved Context"):
                            retrieved_docs = retriever.retrieve(user_question) if hasattr(retriever, 'retrieve') else []
                            for i, doc in enumerate(retrieved_docs[:3]):  # Show top 3
                                st.markdown(f"**Document {i+1}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.divider()
                        
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    logger.error(f"Q&A error: {e}")
    
    # Tab 3: Summary Mode
    with tab3:
        st.header("📝 Summary Mode")
        st.session_state.current_mode = "summary"
        
        if not st.session_state.document_manager.get_active_set():
            st.warning("Please load and index documents first.")
        else:
            # Show active document set info
            active_set = st.session_state.document_manager.get_active_set()
            st.info(f"Active Document Set: {active_set.name} ({active_set.document_count} documents)")
            
            # Summary interface
            summary_query = st.text_area(
                "Enter content to summarize or leave blank to summarize the entire document set:",
                height=100
            )
            
            if st.button("Generate Summary"):
                try:
                    with st.spinner("Generating summary..."):
                        # Get active FAISS index
                        faiss_index = st.session_state.document_manager.get_active_faiss_index()
                        if not faiss_index:
                            st.error("No FAISS index available for the active document set.")
                            return
                        
                        # Get retriever
                        retriever = get_retriever(faiss_index)
                        
                        # Get LLM and build chain
                        llm = get_llm(llm_model, temperature)
                        prompt = PromptTemplates.get_summary_template()
                        
                        if summary_query:
                            # Summarize specific content
                            rag_chain = build_rag_chain(retriever, llm, prompt, "summary")
                            summary = rag_chain.invoke({"question": summary_query})
                        else:
                            # Summarize entire document set
                            all_docs = active_set.documents
                            if all_docs:
                                # Combine all documents for summary
                                combined_content = "\n\n".join([doc.page_content for doc in all_docs[:10]])  # Limit to first 10 docs
                                summary = llm.invoke(prompt.format(context=combined_content))
                            else:
                                st.error("No documents available for summarization.")
                                return
                        
                        # Display summary
                        st.markdown("### Summary")
                        st.write(summary)
                        
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
                    logger.error(f"Summary error: {e}")
    
    # Tab 4: Analysis Mode
    with tab4:
        st.header("📊 Analysis Mode")
        st.session_state.current_mode = "analysis"
        
        if not st.session_state.document_manager.get_active_set():
            st.warning("Please load and index documents first.")
        else:
            # Show active document set info
            active_set = st.session_state.document_manager.get_active_set()
            st.info(f"Active Document Set: {active_set.name} ({active_set.document_count} documents)")
            
            # Analysis interface
            analysis_query = st.text_input("What would you like me to analyze?")
            
            if st.button("Perform Analysis") and analysis_query:
                try:
                    with st.spinner("Performing analysis..."):
                        # Get active FAISS index
                        faiss_index = st.session_state.document_manager.get_active_faiss_index()
                        if not faiss_index:
                            st.error("No FAISS index available for the active document set.")
                            return
                        
                        # Get retriever
                        retriever = get_retriever(faiss_index)
                        
                        # Apply reranking if enabled
                        if st.session_state.reranker_enabled:
                            embeddings = get_embedding_model()
                            reranker = get_default_reranker(embeddings)
                            
                            # Get initial retrieval
                            initial_docs = retriever.retrieve(analysis_query)
                            
                            # Apply reranking
                            reranked_docs = reranker.rerank(analysis_query, initial_docs)
                            
                            # Update retriever with reranked results
                            class RerankedRetriever:
                                def __init__(self, docs):
                                    self.docs = docs
                                
                                def retrieve(self, query):
                                    return self.docs
                            
                            retriever = RerankedRetriever(reranked_docs)
                        
                        # Get LLM and build chain
                        llm = get_llm(llm_model, temperature)
                        prompt = PromptTemplates.get_analysis_template()
                        rag_chain = build_rag_chain(retriever, llm, prompt, "analysis")
                        
                        # Generate analysis
                        analysis = rag_chain.invoke({"question": analysis_query})
                        
                        # Display analysis
                        st.markdown("### Analysis")
                        st.write(analysis)
                        
                        # Show retrieved context
                        with st.expander("View Analysis Context"):
                            retrieved_docs = retriever.retrieve(analysis_query) if hasattr(retriever, 'retrieve') else []
                            for i, doc in enumerate(retrieved_docs[:3]):  # Show top 3
                                st.markdown(f"**Document {i+1}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.divider()
                        
                except Exception as e:
                    st.error(f"Error performing analysis: {e}")
                    logger.error(f"Analysis error: {e}")
    
    # Footer with system information
    st.markdown("---")
    st.markdown("### System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Document Sets:**")
        if available_sets:
            for set_info in available_sets:
                st.text(f"• {set_info['name']}: {set_info['document_count']} docs")
        else:
            st.text("No document sets")
    
    with col2:
        st.markdown("**Current Mode:**")
        st.text(st.session_state.current_mode.upper())
        
        st.markdown("**Reranker:**")
        st.text("Enabled" if st.session_state.reranker_enabled else "Disabled")
    
    with col3:
        st.markdown("**Configuration:**")
        st.text(f"Chunk Size: {st.session_state.chunk_size}")
        st.text(f"Chunk Overlap: {st.session_state.chunk_overlap}")
        st.text(f"Temperature: {temperature}")

if __name__ == "__main__":
    main()
