# Handles document loading and chunking

from langchain_community.document_loaders import ArxivLoader, WikipediaLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_arxiv_paper(query: str) -> List:
    """Load papers from Arxiv using a search query or paper ID."""
    try:
        loader = ArxivLoader(query)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} Arxiv documents for query: {query}")
        return documents
    except Exception as e:
        logger.error(f"Error loading Arxiv paper: {e}")
        raise Exception(f"Failed to load Arxiv paper: {e}")

def load_wikipedia_page(query: str) -> List:
    """Load a Wikipedia page by title or search."""
    try:
        loader = WikipediaLoader(query)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} Wikipedia documents for query: {query}")
        return documents
    except Exception as e:
        logger.error(f"Error loading Wikipedia page: {e}")
        raise Exception(f"Failed to load Wikipedia page: {e}")

def load_web_url(url: str) -> List:
    """Load content from a general web URL."""
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} web documents from URL: {url}")
        return documents
    except Exception as e:
        logger.error(f"Error loading web URL: {e}")
        raise Exception(f"Failed to load web URL: {e}")

def chunk_documents(documents: List, chunk_size: int = 500, chunk_overlap: int = 50) -> List:
    """Split documents into chunks with specified size and overlap."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Successfully chunked {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        raise Exception(f"Failed to chunk documents: {e}")

def validate_source_input(source: str, query: str) -> bool:
    """Validate the source and query inputs."""
    if not query or not query.strip():
        return False
    
    if source == "Web URL" and not (query.startswith('http://') or query.startswith('https://')):
        return False
    
    return True
