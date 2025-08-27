# Utility functions and helpers for the RAG system
import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

# Set up logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for the RAG system."""
    
    DEFAULT_CONFIG = {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "llama3-8b-8192",
        "temperature": 0.1,
        "max_tokens": 2048,
        "retrieval_k": 6,
        "fetch_k": 10,
        "index_name": "research_assistant_index"
    }
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from: {config_path}")
                return {**cls.DEFAULT_CONFIG, **config}
            except Exception as e:
                logger.warning(f"Error loading config file, using defaults: {e}")
        
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], config_path: str = "config.json"):
        """Save configuration to file."""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

def validate_environment() -> Dict[str, bool]:
    """Validate that all required environment variables are set."""
    required_vars = ["GROQ_API_KEY"]
    optional_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    
    validation_results = {}
    
    # Check required variables
    for var in required_vars:
        validation_results[var] = bool(os.getenv(var))
        if not validation_results[var]:
            logger.error(f"Required environment variable not set: {var}")
    
    # Check optional variables
    for var in optional_vars:
        validation_results[var] = bool(os.getenv(var))
        if validation_results[var]:
            logger.info(f"Optional environment variable set: {var}")
    
    return validation_results

def format_timestamp() -> str:
    """Get formatted timestamp for logging and file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized

def chunk_text_by_tokens(text: str, max_tokens: int = 500) -> List[str]:
    """Split text into chunks based on token count."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    except ImportError:
        logger.warning("tiktoken not available, falling back to character-based chunking")
        # Fallback to character-based chunking
        return [text[i:i + max_tokens] for i in range(0, len(text), max_tokens)]

def calculate_similarity_score(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    try:
        import numpy as np
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def get_file_extension(file_path: str) -> str:
    """Get file extension from file path."""
    return os.path.splitext(file_path)[1].lower()

def is_supported_file_type(file_path: str) -> bool:
    """Check if file type is supported for processing."""
    supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.html', '.xml'}
    return get_file_extension(file_path) in supported_extensions

def create_directory_if_not_exists(directory_path: str) -> bool:
    """Create directory if it doesn't exist."""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
            return True
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size: {e}")
        return 0.0
