"""Ollama local LLM generation with streaming support."""

import logging
import time
from typing import Dict, List, Tuple

from ollama import Client

from config.config import RAGConfig, RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def ollama_generate(query: str, context_chunks: List[Dict], chat_history: List[Tuple[str, str]], config: RAGConfig):
    """Generates a streaming response using Ollama local models with retrieved context.

    Args:
        query: The user's question.
        context_chunks: List of relevant chunk dicts from reranking.
        chat_history: List of (user_msg, assistant_msg) tuples.
        config: RAG configuration.

    Yields:
        Text chunks from the streaming response.
    """
    client = Client(host=config.ollama_host)

    # Build context from retrieved chunks (limit to top 5)
    context = "\n\n---\n\n".join([
        f"[Source: {chunk['doc_name']}]\n{chunk['content']}"
        for chunk in context_chunks[:5]
    ])

    system_prompt = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}"

    # Build message history
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": query})

    start = time.time()
    response = client.chat(
        model=config.llm_model,
        messages=messages,
        stream=True,
        think=False,
        options={
            "temperature": 0,
            "num_predict": 256
        }
    )
    print(f"Time until generation starts: {time.time()-start:.2f}s")

    for chunk in response:
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']
