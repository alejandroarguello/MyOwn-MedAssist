#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for RAG using LlamaIndex and FAISS:
1. Load vector store
2. Create query engine
3. Handle retrieval and generation
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FAISS_STORE_DIR = Path("./faiss_store")
EMBED_MODEL = "text-embedding-3-small"  # OpenAI embedding model

class RAGQueryEngine:
    """Query engine for RAG using LlamaIndex and FAISS."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize RAG query engine.
        
        Args:
            model_name: Name of the model to use for generation.
                Can be a fine-tuned model ID.
        """
        self.model_name = model_name
        self.index = None
        self.query_engine = None
        
        # Load index if available
        self._load_index()
    
    def _load_index(self):
        """Load FAISS index if available."""
        if not FAISS_STORE_DIR.exists():
            logger.warning(f"FAISS store not found at {FAISS_STORE_DIR}")
            logger.warning("Please run build_vector_store.py first")
            return
        
        try:
            # Load storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=str(FAISS_STORE_DIR)
            )
            
            # Load index
            self.index = load_index_from_storage(storage_context)
            
            logger.info(f"Loaded FAISS index from {FAISS_STORE_DIR}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
    
    def _create_query_engine(self):
        """Create query engine for RAG."""
        if not self.index:
            logger.error("Index not loaded. Cannot create query engine.")
            return
        
        # Create LLM
        llm = OpenAI(model=self.model_name, temperature=0.1)
        
        # Create embedding model
        embed_model = OpenAIEmbedding(model=EMBED_MODEL)
        
        # Create retriever with parameters
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,  # Number of documents to retrieve
            embed_model=embed_model
        )
        
        # Create postprocessor for filtering results
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            llm=llm
        )
        
        logger.info("Created RAG query engine")
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            query_text: The query text.
            
        Returns:
            Dict containing response and source information.
        """
        if not self.query_engine:
            self._create_query_engine()
            
            if not self.query_engine:
                return {
                    "response": "Error: Query engine not available. Please build the vector store first.",
                    "sources": []
                }
        
        try:
            # Execute query
            response = self.query_engine.query(query_text)
            
            # Extract source nodes
            source_nodes = response.source_nodes
            sources = []
            
            for node in source_nodes:
                sources.append({
                    "text": node.node.text,
                    "score": node.score,
                    "source": node.node.metadata.get("source", "Unknown")
                })
            
            return {
                "response": str(response),
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                "response": f"Error: {str(e)}",
                "sources": []
            }
    
    def query_with_streaming(self, query_text: str, callback=None):
        """Query with streaming response.
        
        Args:
            query_text: The query text.
            callback: Callback function for streaming tokens.
        
        Yields:
            Response tokens as they are generated.
        """
        if not self.query_engine:
            self._create_query_engine()
            
            if not self.query_engine:
                yield "Error: Query engine not available. Please build the vector store first."
                return
        
        try:
            # Set streaming mode
            self.query_engine.llm.streaming = True
            
            # Execute streaming query
            streaming_response = self.query_engine.query(query_text, streaming=True)
            
            # Stream the response
            for token in streaming_response.response_gen:
                if callback:
                    callback(token)
                yield token
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"Error: {str(e)}"

# Singleton instance
_query_engine = None

def get_query_engine(model_name: str = "gpt-3.5-turbo") -> RAGQueryEngine:
    """Get or create a RAG query engine.
    
    Args:
        model_name: Name of the model to use for generation.
    
    Returns:
        RAGQueryEngine instance.
    """
    global _query_engine
    
    if _query_engine is None or _query_engine.model_name != model_name:
        _query_engine = RAGQueryEngine(model_name=model_name)
    
    return _query_engine
