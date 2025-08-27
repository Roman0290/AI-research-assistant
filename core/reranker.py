# Reranker module for improving retrieval quality
from typing import List, Tuple, Dict, Any
from langchain.schema import Document
import logging

# Set up logging
logger = logging.getLogger(__name__)

class BaseReranker:
    """Base class for rerankers."""
    
    def rerank(self, query: str, documents: List[Document], **kwargs) -> List[Document]:
        """Rerank documents based on query relevance."""
        raise NotImplementedError("Subclasses must implement rerank method")

class SimilarityReranker(BaseReranker):
    """Reranker based on embedding similarity scores."""
    
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
    
    def rerank(self, query: str, documents: List[Document], **kwargs) -> List[Document]:
        """Rerank documents by embedding similarity to query."""
        try:
            if not documents:
                return []
            
            # Get query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Get document embeddings
            doc_embeddings = self.embeddings_model.embed_documents(
                [doc.page_content for doc in documents]
            )
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((similarity, documents[i]))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Return reranked documents
            reranked_docs = [doc for _, doc in similarities]
            logger.info(f"Reranked {len(documents)} documents using similarity scores")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in similarity reranking: {e}")
            return documents  # Return original order if reranking fails
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

class KeywordReranker(BaseReranker):
    """Reranker based on keyword matching and TF-IDF scores."""
    
    def __init__(self):
        self.keyword_weights = {}
    
    def rerank(self, query: str, documents: List[Document], **kwargs) -> List[Document]:
        """Rerank documents by keyword relevance to query."""
        try:
            if not documents:
                return []
            
            # Extract keywords from query
            query_keywords = self._extract_keywords(query.lower())
            
            # Calculate keyword scores for each document
            doc_scores = []
            for doc in documents:
                doc_text = doc.page_content.lower()
                doc_keywords = self._extract_keywords(doc_text)
                
                # Calculate keyword overlap score
                overlap_score = len(query_keywords.intersection(doc_keywords))
                
                # Calculate TF-IDF like score
                tfidf_score = self._calculate_tfidf_score(query_keywords, doc_text)
                
                # Combined score
                total_score = overlap_score + tfidf_score
                doc_scores.append((total_score, doc))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Return reranked documents
            reranked_docs = [doc for _, doc in doc_scores]
            logger.info(f"Reranked {len(documents)} documents using keyword matching")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in keyword reranking: {e}")
            return documents  # Return original order if reranking fails
    
    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text."""
        import re
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words (alphanumeric + hyphens)
        words = re.findall(r'\b[a-zA-Z0-9-]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = {word for word in words if word not in stop_words and len(word) > 2}
        
        return keywords
    
    def _calculate_tfidf_score(self, query_keywords: set, doc_text: str) -> float:
        """Calculate a simple TF-IDF like score."""
        if not query_keywords:
            return 0.0
        
        score = 0.0
        doc_words = doc_text.split()
        total_words = len(doc_words)
        
        if total_words == 0:
            return 0.0
        
        for keyword in query_keywords:
            # Count occurrences of keyword in document
            keyword_count = doc_text.count(keyword.lower())
            
            # Simple TF-IDF calculation
            tf = keyword_count / total_words
            score += tf
        
        return score

class HybridReranker(BaseReranker):
    """Combines multiple reranking strategies."""
    
    def __init__(self, rerankers: List[BaseReranker], weights: List[float] = None):
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)
        
        if len(self.weights) != len(self.rerankers):
            raise ValueError("Number of weights must match number of rerankers")
    
    def rerank(self, query: str, documents: List[Document], **kwargs) -> List[Document]:
        """Rerank documents using multiple strategies."""
        try:
            if not documents:
                return []
            
            # Get reranked documents from each reranker
            reranked_results = []
            for reranker in self.rerankers:
                reranked_docs = reranker.rerank(query, documents, **kwargs)
                reranked_results.append(reranked_docs)
            
            # Combine scores using weighted voting
            doc_scores = {}
            for i, reranked_docs in enumerate(reranked_results):
                weight = self.weights[i]
                for j, doc in enumerate(reranked_docs):
                    if doc not in doc_scores:
                        doc_scores[doc] = 0.0
                    # Higher rank = higher score
                    doc_scores[doc] += weight * (len(documents) - j)
            
            # Sort by combined score
            scored_docs = [(score, doc) for doc, score in doc_scores.items()]
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Return reranked documents
            reranked_docs = [doc for _, doc in scored_docs]
            logger.info(f"Reranked {len(documents)} documents using hybrid approach")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in hybrid reranking: {e}")
            return documents  # Return original order if reranking fails

def get_default_reranker(embeddings_model=None) -> BaseReranker:
    """Get a default reranker configuration."""
    if embeddings_model:
        return SimilarityReranker(embeddings_model)
    else:
        return KeywordReranker()
