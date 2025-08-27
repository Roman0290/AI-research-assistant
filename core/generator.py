# LLM and prompt templating logic
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, Optional, List
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

class PromptTemplates:
    """Collection of prompt templates for different use cases."""
    
    @staticmethod
    def get_qa_template() -> PromptTemplate:
        """Get the main Q&A prompt template."""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI research assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so
- Be concise but comprehensive
- Use clear, academic language
- Cite specific parts of the context when relevant

Answer:"""
        )
    
    @staticmethod
    def get_summary_template() -> PromptTemplate:
        """Get the summary prompt template."""
        return PromptTemplate(
            input_variables=["context"],
            template="""You are an AI research assistant. Provide a comprehensive summary of the following content.

Content:
{context}

Instructions:
- Create a structured summary with key points
- Identify main themes and findings
- Highlight important conclusions
- Use clear, academic language
- Organize information logically

Summary:"""
        )
    
    @staticmethod
    def get_analysis_template() -> PromptTemplate:
        """Get the analysis prompt template."""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI research assistant. Analyze the following content based on the user's question.

Content:
{context}

Analysis Request: {question}

Instructions:
- Provide a detailed analysis based on the content
- Identify patterns, trends, or insights
- Compare and contrast different aspects if relevant
- Draw conclusions based on the evidence
- Use analytical and critical thinking

Analysis:"""
        )

def get_llm(model_name: str = "llama3-8b-8192", temperature: float = 0.1) -> ChatGroq:
    """Get the Groq LLM instance with configuration."""
    try:
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=2048
        )
        logger.info(f"Successfully initialized Groq LLM with model: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Error initializing Groq LLM: {e}")
        raise Exception(f"Failed to initialize Groq LLM: {e}")

def build_rag_chain(retriever, llm, prompt: PromptTemplate, mode: str = "qa") -> Any:
    """Build a RAG chain for different generation modes."""
    
    def format_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs based on the generation mode."""
        try:
            if mode == "qa":
                # For Q&A, retrieve relevant docs and format for answer generation
                docs = retriever.retrieve(inputs["question"])
                context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                     for i, doc in enumerate(docs)])
                return {"context": context, "question": inputs["question"]}
            
            elif mode == "summary":
                # For summary, use the question as context
                return {"context": inputs["question"]}
            
            elif mode == "analysis":
                # For analysis, retrieve docs and format for analysis
                docs = retriever.retrieve(inputs["question"])
                context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" 
                                     for i, doc in enumerate(docs)])
                return {"context": context, "question": inputs["question"]}
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            logger.error(f"Error formatting inputs: {e}")
            return {"context": "Error retrieving context", "question": inputs.get("question", "")}
    
    try:
        chain = RunnablePassthrough() | RunnableLambda(format_inputs) | prompt | llm | StrOutputParser()
        logger.info(f"Successfully built RAG chain for mode: {mode}")
        return chain
    except Exception as e:
        logger.error(f"Error building RAG chain: {e}")
        raise Exception(f"Failed to build RAG chain: {e}")

def get_available_models() -> List[str]:
    """Get list of available Groq models."""
    return [
        "llama3-8b-8192",
        "llama3-70b-8192", 
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama2-70b-4096"
    ]

def validate_llm_response(response: Any) -> str:
    """Validate and extract content from LLM response."""
    try:
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        logger.error(f"Error validating LLM response: {e}")
        return "Error processing response"
