"""General knowledge fallback mode (no RAG context)."""

import logging

from ollama import Client
import streamlit as st 

from config.config import RAGConfig

logger = logging.getLogger(__name__)


def handle_fallback(query: str, config: RAGConfig) -> str:
    """Generates a response using Ollama without document context.

    Used when no relevant documents are found or confidence is too low.

    Args:
        query: The user's question.
        config: RAG configuration.

    Returns:
        The generated response text.
    """
    try:
        client = Client(host=config.ollama_host)
        system_prompt = """You are a helpful AI assistant. When you don't know something, 
        be honest about it. Provide clear, concise, and accurate responses. If the question 
        is not related to any specific document, use your general knowledge to answer."""

        response = client.chat(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            think=False
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Fallback error: {str(e)}")
        st.error(f"Fallback error: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."
