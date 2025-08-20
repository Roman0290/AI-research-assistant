# AI-research-assistant-rag

## 📖 Overview
**AI-research-assistant-rag** is a Retrieval-Augmented Generation (RAG) system built with **LangChain**.  
It combines Large Language Models (LLMs) with external sources like **Arxiv, Wikipedia, and the Web** to provide accurate, context-aware answers for research-related questions.

---

## ⚡ Features
- 🔍 Load research papers (Arxiv), Wikipedia, or web content  
- 📑 Chunk & embed text into vector representations (FAISS)  
- 🤖 Retrieve relevant docs and generate answers with LLMs  
- 🧠 Extensible for more sources (PDFs, PubMed, etc.)

---

## ⚙️ Installation
```bash
git clone https://github.com/your-username/AI-research-assistant-rag.git
cd AI-research-assistant-rag
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate
pip install -r requirements.txt
