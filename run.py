#!/usr/bin/env python3
"""
Startup script for AI Research Assistant RAG system.
This script handles environment setup and launches the Streamlit application.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required. Current version: %s", sys.version)
        return False
    logger.info("✅ Python version check passed: %s", sys.version)
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = {
        'streamlit': 'streamlit',
        'langchain': 'langchain',
        'faiss-cpu': 'faiss',  
        'sentence-transformers': 'sentence_transformers',
        'langchain-groq': 'langchain_groq',
        'arxiv': 'arxiv',
        'wikipedia': 'wikipedia',
        'requests': 'requests'
    }

    missing_packages = []

    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"⚠️ {package} is not installed")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True

def check_environment():
    """Check environment variables."""
    groq_key = os.getenv('GROQ_API_KEY')
    
    if not groq_key:
        logger.warning("⚠️ GROQ_API_KEY environment variable is not set")
        logger.info("Please set your Groq API key:")
        logger.info("  export GROQ_API_KEY=your_api_key_here")
        logger.info("  or create a .env file with GROQ_API_KEY=your_api_key_here")
        return False
    
    logger.info("✅ GROQ_API_KEY is configured")
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['document_sets', 'logs', 'temp']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory ready: {directory}")

def run_streamlit():
    """Run the Streamlit application."""
    try:
        app_path = Path("app/main.py")
        
        if not app_path.exists():
            logger.error(f"❌ Application file not found: {app_path}")
            return False
        
        logger.info("🚀 Starting AI Research Assistant...")
        logger.info("📱 The application will open in your browser")
        logger.info("🔗 URL: http://localhost:8501")
        logger.info("⏹️  Press Ctrl+C to stop the application")
        
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Application stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Error running Streamlit: {e}")
        return False

def main():
    """Main startup function."""
    logger.info("🤖 AI Research Assistant - System Startup")
    logger.info("=" * 50)
    
    # System checks
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Environment check (warning only, not fatal)
    env_ok = check_environment()
    if not env_ok:
        logger.warning("⚠️ Environment not fully configured, but continuing...")
    
    # Create directories
    create_directories()
    
    # Run application
    success = run_streamlit()
    
    if success:
        logger.info("✅ Application completed successfully")
        return 0
    else:
        logger.error("❌ Application failed to start")
        return 1

if __name__ == "__main__":
    sys.exit(main())
