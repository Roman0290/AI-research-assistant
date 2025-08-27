# Vector store and retrieval logic
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """Enhanced retriever with reranking and multiple search strategies."""
    
    def __init__(self, faiss_index, search_kwargs: Optional[Dict[str, Any]] = None):
        self.faiss_index = faiss_index
        self.search_kwargs = search_kwargs or {"k": 6}
        self.retriever = faiss_index.as_retriever(search_kwargs=self.search_kwargs)
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve documents using the configured retriever."""
        try:
            search_k = k or self.search_kwargs.get("k", 6)
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs[:search_k]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Retrieve documents with similarity scores."""
        try:
            search_k = k or self.search_kwargs.get("k", 6)
            docs_and_scores = self.faiss_index.similarity_search_with_score(query, k=search_k)
            logger.info(f"Retrieved {len(docs_and_scores)} documents with scores for query: {query[:50]}...")
            return docs_and_scores
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {e}")
            return []

def build_faiss_index(embeddings, documents: List[Document], 
                      index_name: str = "research_assistant_index") -> FAISS:
    """Create a FAISS vector store from documents and embeddings."""
    try:
        # Check if index already exists
        if os.path.exists(f"{index_name}.pkl") and os.path.exists(f"{index_name}.faiss"):
            logger.info(f"Loading existing FAISS index: {index_name}")
            return FAISS.load_local(index_name, embeddings)
        
        # Create new index
        logger.info(f"Creating new FAISS index with {len(documents)} documents")
        faiss_index = FAISS.from_documents(documents, embeddings)
        
        # Save the index for future use
        faiss_index.save_local(index_name)
        logger.info(f"FAISS index saved as: {index_name}")
        
        return faiss_index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise Exception(f"Failed to build FAISS index: {e}")

def get_retriever(faiss_index, search_kwargs: Optional[Dict[str, Any]] = None):
    """Return an enhanced retriever for dense retrieval from FAISS."""
    default_kwargs = {"k": 6, "fetch_k": 10}
    if search_kwargs:
        default_kwargs.update(search_kwargs)
    
    return EnhancedRetriever(faiss_index, default_kwargs)

def add_documents_to_index(faiss_index: FAISS, new_documents: List[Document], 
                          embeddings) -> FAISS:
    """Add new documents to an existing FAISS index."""
    try:
        faiss_index.add_documents(new_documents)
        logger.info(f"Added {len(new_documents)} new documents to existing index")
        return faiss_index
    except Exception as e:
        logger.error(f"Error adding documents to index: {e}")
        raise Exception(f"Failed to add documents to index: {e}")

def get_index_stats(faiss_index: FAISS) -> Dict[str, Any]:
    """Get statistics about the FAISS index."""
    try:
        return {
            "total_documents": faiss_index.index.ntotal if hasattr(faiss_index.index, 'ntotal') else "Unknown",
            "index_type": type(faiss_index.index).__name__,
            "embedding_dimension": faiss_index.index.d if hasattr(faiss_index.index, 'd') else "Unknown"
        }
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return {"error": str(e)}
