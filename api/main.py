#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastAPI application for MyOwn-MedAssist:
1. Provides async, token-streaming endpoints
2. Integrates with fine-tuned model and RAG
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import streaming utilities
from api.streaming_utils import stream_openai_response, create_streaming_response

# Try to import RAG utilities, with fallback if not available
try:
    from rag.llama_index_utils import get_query_engine
    RAG_AVAILABLE = True
except ImportError:
    logger.warning("LlamaIndex utilities not available. RAG functionality will be disabled.")
    RAG_AVAILABLE = False
    
    # Define mock RAG functions for fallback
    def get_query_engine(model_name=None):
        class MockQueryEngine:
            def query(self, query_text):
                return {
                    "response": f"[Mock RAG Response] For query: {query_text}\n\nThis is a simulated response since the RAG system is not available. Please ensure LlamaIndex is properly installed and the vector store is built.",
                    "sources": [
                        {"text": "This is a mock source reference.", "source": "Mock Medical Journal", "url": "#"}
                    ]
                }
                
            def query_with_streaming(self, query_text):
                response = self.query(query_text)["response"]
                words = response.split()
                for word in words:
                    yield word + " "
                    
        return MockQueryEngine()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_INFO_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fine_tuning", "models", "model_info.json")
DEFAULT_MODEL = "gpt-3.5-turbo"  # Default model if fine-tuned model not available

# Ensure OpenAI API key is available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables. API calls will fail.")

# Create a demo fine-tuned model ID if none exists (for testing)
demo_fine_tuned_id = "ft:gpt-3.5-turbo-0125:medical-qa:1234567890"

# Load fine-tuned model info if available
fine_tuned_model = None
try:
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, "r") as f:
            model_info = json.load(f)
        # Check for both possible keys in the model info file
        fine_tuned_model = model_info.get("fine_tuned_model") or model_info.get("model_id")
        if fine_tuned_model:
            logger.info(f"Loaded fine-tuned model: {fine_tuned_model}")
        else:
            # If neither key exists, use the demo model
            fine_tuned_model = demo_fine_tuned_id
            logger.warning(f"No model ID found in model info file. Using demo model ID: {fine_tuned_model}")
    else:
        # If model info doesn't exist, create directory and use demo model
        os.makedirs(os.path.dirname(MODEL_INFO_PATH), exist_ok=True)
        fine_tuned_model = demo_fine_tuned_id
        logger.warning(f"Model info file not found. Using demo model ID: {fine_tuned_model}")
        # Save demo model info for future use
        with open(MODEL_INFO_PATH, "w") as f:
            json.dump({"model_id": fine_tuned_model}, f)
except Exception as e:
    logger.warning(f"Error loading fine-tuned model info: {e}")
    fine_tuned_model = demo_fine_tuned_id

# Initialize FastAPI app
app = FastAPI(
    title="MyOwn-MedAssist API",
    description="API for medical question answering with fine-tuned model and RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    model: Optional[str] = Field(None, description="Model to use (default: fine-tuned if available, otherwise gpt-3.5-turbo)")
    use_rag: bool = Field(False, description="Whether to use RAG for the response")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: float = Field(0.7, description="Temperature for response generation")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Model response")
    model: str = Field(..., description="Model used for the response")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Sources used for RAG (if enabled)")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MyOwn-MedAssist API",
        "docs": "/docs",
        "models": {
            "default": DEFAULT_MODEL,
            "fine_tuned": fine_tuned_model
        }
    }

@app.get("/models")
async def get_models():
    """Get available models."""
    models = [DEFAULT_MODEL]
    if fine_tuned_model:
        models.append(fine_tuned_model)
    
    return {
        "models": models,
        "default": fine_tuned_model or DEFAULT_MODEL
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for non-streaming responses."""
    # Determine which model to use
    model = request.model or fine_tuned_model or DEFAULT_MODEL
    
    # Convert Pydantic messages to dict format for OpenAI
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    if request.use_rag:
        # Extract the user's query from the last user message
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1]["content"]
        
        # Get RAG response
        rag_engine = get_query_engine(model_name=model)
        result = rag_engine.query(query)
        
        return ChatResponse(
            response=result["response"],
            model=f"{model} with RAG",
            sources=result.get("sources")
        )
    else:
        # Use OpenAI API directly
        from openai import OpenAI
        
        # Initialize OpenAI client with API key from environment
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        try:
            # Check if API key is available
            if not os.getenv("OPENAI_API_KEY"):
                # Return mock response for testing when no API key is available
                logger.warning("No OpenAI API key found. Returning mock response.")
                mock_response = "This is a mock response from the Medical QA system since no OpenAI API key is available. "
                mock_response += "In a real deployment, this would be a response from the model addressing your medical question. "
                mock_response += "Please ensure you have set the OPENAI_API_KEY environment variable to use the actual model."
                
                return ChatResponse(
                    response=mock_response,
                    model=f"Mock {model}",
                    sources=None
                )
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature
            )
            
            return ChatResponse(
                response=response.choices[0].message.content,
                model=model,
                sources=None
            )
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Chat endpoint with streaming response."""
    # Determine which model to use
    model = request.model or fine_tuned_model or DEFAULT_MODEL
    
    # Convert Pydantic messages to dict format for OpenAI
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    if request.use_rag:
        # Extract the user's query from the last user message
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1]["content"]
        
        # Get streaming RAG response
        rag_engine = get_query_engine(model_name=model)
        
        async def stream_rag():
            # Create header with model info
            yield json.dumps({"model": f"{model} with RAG"}) + "\n"
            
            # Stream the response
            for token in rag_engine.query_with_streaming(query):
                yield json.dumps({"token": token}) + "\n"
            
            # End with sources if available
            try:
                result = rag_engine.query(query)
                if "sources" in result and result["sources"]:
                    yield json.dumps({"sources": result["sources"]}) + "\n"
            except Exception as e:
                logger.error(f"Error getting RAG sources: {e}")
        
        return StreamingResponse(
            stream_rag(),
            media_type="text/event-stream"
        )
    else:
        # Use OpenAI API with streaming
        # Check if API key is available
        if not os.getenv("OPENAI_API_KEY"):
            # Return mock streaming response for testing when no API key is available
            logger.warning("No OpenAI API key found. Returning mock streaming response.")
            
            async def mock_stream():
                # Send model info
                yield json.dumps({"model": f"Mock {model}"}) + "\n"
                
                # Send mock response word by word
                mock_text = "This is a mock response from the Medical QA system since no OpenAI API key is available. "
                mock_text += "In a real deployment, this would be a streaming response from the model addressing your medical question. "
                mock_text += "Please ensure you have set the OPENAI_API_KEY environment variable to use the actual model."
                
                words = mock_text.split()
                for word in words:
                    yield json.dumps({"token": word + " "}) + "\n"
                    await asyncio.sleep(0.05)  # Simulate realistic typing speed
            
            return StreamingResponse(
                mock_stream(),
                media_type="text/event-stream"
            )
        
        # Call the actual streaming function if API key exists
        return await stream_openai_response(messages, model, request.temperature)

if __name__ == "__main__":
    import uvicorn
    
    # Log information about the API configuration
    logger.info(f"Starting MyOwn-MedAssist API with the following configuration:")
    logger.info(f"- Default model: {DEFAULT_MODEL}")
    logger.info(f"- Fine-tuned model: {fine_tuned_model}")
    logger.info(f"- OpenAI API key available: {bool(os.getenv('OPENAI_API_KEY'))}")
    logger.info(f"- RAG functionality available: {RAG_AVAILABLE}")
    
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
