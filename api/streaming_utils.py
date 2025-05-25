#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for token streaming in the FastAPI application.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def stream_openai_response(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7
) -> StreamingResponse:
    """Stream response from OpenAI API.
    
    Args:
        messages: List of messages in the conversation
        model: Model to use
        temperature: Temperature for response generation
        
    Returns:
        StreamingResponse with the model's response
    """
    client = AsyncOpenAI()
    
    async def generate():
        try:
            # Send model info first
            yield json.dumps({"model": model}) + "\n"
            
            # Create streaming response
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield json.dumps({"token": content}) + "\n"
        
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield json.dumps({"error": str(e)}) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

def create_streaming_response(
    generator_func: AsyncGenerator
) -> StreamingResponse:
    """Create a streaming response from an async generator.
    
    Args:
        generator_func: Async generator function that yields tokens
        
    Returns:
        StreamingResponse with the streaming content
    """
    return StreamingResponse(
        generator_func,
        media_type="text/event-stream"
    )

async def stream_text(text: str):
    """Stream a text string token by token (for testing).
    
    Args:
        text: Text to stream
        
    Yields:
        JSON-encoded tokens
    """
    # Send a header
    yield json.dumps({"model": "test-model"}) + "\n"
    
    # Stream the text word by word
    words = text.split()
    for word in words:
        yield json.dumps({"token": word + " "}) + "\n"
        await asyncio.sleep(0.1)  # Simulate delay
