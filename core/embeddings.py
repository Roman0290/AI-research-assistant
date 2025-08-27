# Embedding model setup
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                        device: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    Get the embedding model for document vectorization.
    
    Args:
        model_name: The name of the sentence transformer model to use
        device: Device to run the model on ('cpu' or 'cuda')
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device} if device else {},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Successfully loaded embedding model: {model_name}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        raise Exception(f"Failed to load embedding model: {e}")

def get_embedding_dimension(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> int:
    """
    Get the dimension of embeddings for the specified model.
    
    Args:
        model_name: The name of the sentence transformer model
    
    Returns:
        Embedding dimension as integer
    """
    # Common embedding dimensions for popular models
    model_dimensions = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L12-v2": 384,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384
    }
    
    return model_dimensions.get(model_name, 384)  # Default to 384 if unknown
