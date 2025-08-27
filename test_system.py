#!/usr/bin/env python3
"""
Test script for AI Research Assistant RAG system.
This script tests the core functionality without requiring the full Streamlit interface.
"""

import os
import sys
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        from core.ingestion import load_arxiv_paper, load_wikipedia_page, load_web_url, chunk_documents
        from core.embeddings import get_embedding_model
        from core.retriever import build_faiss_index, get_retriever
        from core.generator import PromptTemplates, get_llm, build_rag_chain
        from core.reranker import get_default_reranker
        from core.document_manager import DocumentManager
        from utils.helpers import ConfigManager, validate_environment
        
        logger.info("✅ All core modules imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    logger.info("Testing configuration management...")
    
    try:
        from utils.helpers import ConfigManager
        
        # Test default config
        config = ConfigManager.load_config()
        logger.info(f"✅ Default configuration loaded: {len(config)} parameters")
        
        # Test config saving
        test_config = {"test_param": "test_value"}
        ConfigManager.save_config(test_config, "test_config.json")
        logger.info("✅ Configuration saving works")
        
        # Clean up test file
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_embeddings():
    """Test embedding model functionality."""
    logger.info("Testing embedding model...")
    
    try:
        from core.embeddings import get_embedding_model
        
        embeddings = get_embedding_model()
        logger.info("✅ Embedding model loaded successfully")
        
        # Test embedding generation
        test_text = "This is a test document for embedding generation."
        embedding = embeddings.embed_query(test_text)
        
        if embedding and len(embedding) > 0:
            logger.info(f"✅ Embedding generated successfully: {len(embedding)} dimensions")
            return True
        else:
            logger.error("❌ Embedding generation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        return False

def test_document_processing():
    """Test document processing and chunking."""
    logger.info("Testing document processing...")
    
    try:
        from core.ingestion import chunk_documents
        from langchain.schema import Document
        
        # Create test documents
        test_docs = [
            Document(page_content="This is the first test document with some content to process."),
            Document(page_content="This is the second test document with different content for testing."),
            Document(page_content="This is the third test document to ensure chunking works properly.")
        ]
        
        # Test chunking
        chunks = chunk_documents(test_docs, chunk_size=50, chunk_overlap=10)
        
        if chunks and len(chunks) > len(test_docs):
            logger.info(f"✅ Document chunking successful: {len(test_docs)} docs → {len(chunks)} chunks")
            return True
        else:
            logger.error("❌ Document chunking failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Document processing test failed: {e}")
        return False

def test_document_manager():
    """Test document manager functionality."""
    logger.info("Testing document manager...")
    
    try:
        from core.document_manager import DocumentManager
        
        # Create document manager
        doc_manager = DocumentManager("test_document_sets")
        logger.info("✅ Document manager created successfully")
        
        # Test document set creation
        test_set_name = "test_set"
        doc_manager.create_document_set(test_set_name, "Test document set", "test")
        logger.info("✅ Document set creation successful")
        
        # Test listing document sets
        sets = doc_manager.list_document_sets()
        if sets and len(sets) > 0:
            logger.info(f"✅ Document set listing successful: {len(sets)} sets found")
        else:
            logger.warning("⚠️ No document sets found")
        
        # Clean up test data
        doc_manager.delete_document_set(test_set_name)
        logger.info("✅ Test cleanup successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Document manager test failed: {e}")
        return False

def test_reranker():
    """Test reranker functionality."""
    logger.info("Testing reranker...")
    
    try:
        from core.reranker import get_default_reranker
        from langchain.schema import Document
        
        # Create test documents
        test_docs = [
            Document(page_content="This document contains information about machine learning."),
            Document(page_content="This document discusses artificial intelligence concepts."),
            Document(page_content="This document covers data science topics.")
        ]
        
        # Test keyword reranker (doesn't require embeddings)
        reranker = get_default_reranker()
        
        # Test reranking
        reranked_docs = reranker.rerank("machine learning", test_docs)
        
        if reranked_docs and len(reranked_docs) == len(test_docs):
            logger.info("✅ Reranker functionality successful")
            return True
        else:
            logger.error("❌ Reranker functionality failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Reranker test failed: {e}")
        return False

def test_prompt_templates():
    """Test prompt template functionality."""
    logger.info("Testing prompt templates...")
    
    try:
        from core.generator import PromptTemplates
        
        # Test QA template
        qa_template = PromptTemplates.get_qa_template()
        if qa_template and hasattr(qa_template, 'input_variables'):
            logger.info("✅ QA template loaded successfully")
        
        # Test summary template
        summary_template = PromptTemplates.get_summary_template()
        if summary_template and hasattr(summary_template, 'input_variables'):
            logger.info("✅ Summary template loaded successfully")
        
        # Test analysis template
        analysis_template = PromptTemplates.get_analysis_template()
        if analysis_template and hasattr(analysis_template, 'input_variables'):
            logger.info("✅ Analysis template loaded successfully")
        
        logger.info("✅ All prompt templates working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prompt template test failed: {e}")
        return False

def test_environment_validation():
    """Test environment variable validation."""
    logger.info("Testing environment validation...")
    
    try:
        from utils.helpers import validate_environment
        
        # Test without GROQ_API_KEY
        env_status = validate_environment()
        
        if "GROQ_API_KEY" in env_status:
            if env_status["GROQ_API_KEY"]:
                logger.info("✅ GROQ_API_KEY is set")
            else:
                logger.warning("⚠️ GROQ_API_KEY is not set (this is expected for testing)")
            
            logger.info("✅ Environment validation working correctly")
            return True
        else:
            logger.error("❌ Environment validation failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Environment validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting AI Research Assistant system tests...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration Management", test_configuration),
        ("Embedding Model", test_embeddings),
        ("Document Processing", test_document_processing),
        ("Document Manager", test_document_manager),
        ("Reranker", test_reranker),
        ("Prompt Templates", test_prompt_templates),
        ("Environment Validation", test_environment_validation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
