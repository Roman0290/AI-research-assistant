
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?logo=streamlit)](https://ai-research-assistant-rag.streamlit.app/)

# 🤖 AI Research Assistant (RAG)

A comprehensive Retrieval-Augmented Generation (RAG) system built with **LangChain** that combines Large Language Models (LLMs) with external knowledge sources to provide accurate, context-aware answers for research-related questions.

## 🚀 Features

### Core RAG Capabilities
- **Multi-Source Document Ingestion**: Load research papers from Arxiv, Wikipedia pages, and web content
- **Intelligent Chunking**: Recursive character text splitting with configurable chunk size and overlap
- **Vector Embeddings**: Generate document embeddings using sentence-transformers/all-MiniLM-L6-v2
- **FAISS Vector Store**: Fast and efficient similarity search and retrieval
- **Groq LLM Integration**: High-performance language model for answer generation

### Advanced Features (Stretch Goals)
- **🔄 Reranker System**: Improve retrieval quality with similarity and keyword-based reranking
- **📚 Document Set Management**: Create, manage, and switch between multiple document collections
- **📝 Multiple Generation Modes**: Q&A, summarization, and analysis capabilities
- **⚙️ Configurable Parameters**: Adjustable chunk sizes, temperature, and model selection

### User Interface
- **Streamlit Web App**: Modern, responsive interface with tabbed navigation
- **Real-time Processing**: Live document ingestion and indexing
- **Interactive Configuration**: Adjustable parameters and model settings
- **Document Set Switching**: Seamlessly switch between different knowledge bases

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Vector        │    │   Language      │
│   Sources       │───▶│   Store         │───▶│   Model         │
│                 │    │   (FAISS)       │    │   (Groq)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Retrieval     │    │   Generation    │
│   Processing    │    │   Engine        │    │   Chain         │
│   & Chunking    │    │   (Reranker)    │    │   (RAG)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

- **Framework**: LangChain (Python)
- **LLM**: Groq (llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, etc.)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Web Framework**: Streamlit
- **Document Loaders**: Arxiv, Wikipedia, Web URL
- **Text Processing**: RecursiveCharacterTextSplitter
- **Additional**: tiktoken, numpy, logging

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Internet connection for document loading

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Roman0290/AI-research-assistant.git
cd AI-research-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

Or set it directly in your shell:
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the Application
```bash
streamlit run app/main.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Document Ingestion
1. **Create Document Set**: Use the sidebar to create a new document collection
2. **Choose Source**: Select from Arxiv, Wikipedia, or Web URL
3. **Enter Query**: 
   - Arxiv: Paper ID (e.g., 2303.08774) or search query
   - Wikipedia: Page title or search term
   - Web URL: Valid HTTP/HTTPS URL
4. **Process Documents**: Click "Load & Process Documents" to ingest and index

### 2. Q&A Mode
- Ask questions about your loaded documents
- View retrieved context and generated answers
- Enable reranker for improved retrieval quality

### 3. Summary Mode
- Generate comprehensive summaries of specific content or entire document sets
- Useful for getting overviews of large research papers or collections

### 4. Analysis Mode
- Perform detailed analysis of documents based on specific queries
- Identify patterns, trends, and insights

### 5. Document Set Management
- Create multiple document collections for different research areas
- Switch between sets seamlessly
- Export/import document sets for sharing

## 🔧 Configuration

### Model Settings
- **LLM Model**: Choose from available Groq models
- **Temperature**: Control response creativity (0.0 = deterministic, 1.0 = creative)
- **Chunk Size**: Adjust document chunk size (100-2000 characters)
- **Chunk Overlap**: Set overlap between chunks (0-500 characters)

### Advanced Features
- **Reranker**: Enable similarity and keyword-based reranking
- **Document Sets**: Organize documents into logical collections
- **Persistence**: Automatic saving of document sets and indices

## 📊 Example Queries and Answers

### Arxiv Research Paper
**Query**: "What are the main findings of the paper?"
**Answer**: "Based on the research paper, the main findings include: [detailed analysis of paper content]"

### Wikipedia Article
**Query**: "What is the historical significance of this topic?"
**Answer**: "The historical significance includes: [contextual information from Wikipedia]"

### Web Content
**Query**: "What are the key points discussed?"
**Answer**: "The key points discussed are: [extracted information from web content]"

## 🎯 Use Cases

- **Academic Research**: Analyze research papers and academic literature
- **Content Summarization**: Generate summaries of long documents
- **Knowledge Discovery**: Explore and understand new topics
- **Research Assistance**: Get quick answers from large document collections
- **Content Analysis**: Identify patterns and insights in text data

## ⚠️ Known Limitations

1. **API Rate Limits**: Groq API has rate limits that may affect high-volume usage
2. **Document Size**: Very large documents may require longer processing times
3. **Internet Dependency**: Arxiv and Wikipedia loading requires internet connectivity
4. **Model Context**: LLM responses are limited by token context windows
5. **Embedding Quality**: Retrieval quality depends on embedding model performance

## 🔮 Future Enhancements

- [ ] **Multi-Modal Support**: Handle images, PDFs, and other document types
- [ ] **Advanced Reranking**: Implement more sophisticated reranking algorithms
- [ ] **Collaborative Features**: Share document sets and collaborate with others
- [ ] **Performance Optimization**: Implement caching and batch processing
- [ ] **API Endpoints**: RESTful API for programmatic access
- [ ] **Mobile Support**: Responsive design for mobile devices

## 🐛 Troubleshooting

### Common Issues

1. **GROQ_API_KEY not set**
   - Solution: Set the environment variable or create a `.env` file

2. **Document loading fails**
   - Check internet connectivity
   - Verify input format (Arxiv ID, Wikipedia title, valid URL)
   - Check API rate limits

3. **Memory issues with large documents**
   - Reduce chunk size
   - Process documents in smaller batches
   - Use more efficient embedding models

4. **Slow retrieval**
   - Enable reranker for better quality
   - Adjust chunk overlap settings
   - Consider using smaller embedding models

### Debug Mode
Enable detailed logging by setting the log level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **Groq**: For high-performance LLM access
- **FAISS**: For efficient vector similarity search
- **Streamlit**: For the beautiful web interface
- **Open Source Community**: For the amazing tools and libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Roman0290/AI-research-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Roman0290/AI-research-assistant/discussions)
- **Documentation**: [Wiki](https://github.com/Roman0290/AI-research-assistant/wiki)

---


*Empowering researchers with intelligent document analysis and knowledge discovery.*
