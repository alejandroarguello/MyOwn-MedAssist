#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluation harness for benchmarking models with RAG capabilities.

This module provides functionality to evaluate different model configurations:
1. Baseline vs. fine-tuned models
2. With and without RAG (Retrieval-Augmented Generation)
3. Logging evaluation results to Langfuse
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables first
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
TEST_DATA_PATH = Path("fine_tuning/data/test_small.jsonl")
RESULTS_DIR = Path("./results")
BASELINE_MODEL = "gpt-3.5-turbo"  # Default baseline model

class EvaluationHarness:
    """Evaluation harness for benchmarking models with RAG capabilities."""
    
    def __init__(
        self, 
        baseline_model: str = BASELINE_MODEL,
        fine_tuned_model: Optional[str] = None,
        use_rag: bool = False,
        langfuse_enabled: bool = True,
        limit: Optional[int] = None
    ) -> None:
        """Initialize evaluation harness."""
        self.baseline_model = baseline_model
        self.fine_tuned_model = fine_tuned_model
        self.use_rag = use_rag
        self.langfuse_enabled = langfuse_enabled
        self.limit = limit
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Initialize RAG components if needed
        self.retriever = None
        if use_rag:
            self.retriever = self._initialize_retriever()
        
        # Initialize Langfuse if enabled
        self.langfuse = None
        if langfuse_enabled:
            try:
                from langfuse import Langfuse
                
                # Get Langfuse credentials from environment variables
                public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
                secret_key = os.getenv("LANGFUSE_SECRET_KEY")
                host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                
                if not public_key or not secret_key:
                    logger.warning("Langfuse API keys not found, disabling logging")
                    self.langfuse_enabled = False
                else:
                    # Initialize Langfuse with explicit credentials
                    self.langfuse = Langfuse(
                        public_key=public_key,
                        secret_key=secret_key,
                        host=host
                    )
                    logger.info("Langfuse initialized successfully")
            except ImportError:
                logger.warning("Langfuse not installed, disabling logging")
                self.langfuse_enabled = False
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data from JSONL file."""
        if not TEST_DATA_PATH.exists():
            logger.error("Test data not found: %s", TEST_DATA_PATH)
            return []
        
        try:
            with open(TEST_DATA_PATH, 'r') as f:
                examples = [json.loads(line) for line in f]
            
            # Apply limit if specified
            if self.limit is not None and self.limit > 0:
                logger.info(f"Limiting evaluation to {self.limit} examples (out of {len(examples)} total)")
                examples = examples[:self.limit]
            
            return examples
        except Exception as e:
            logger.error("Error loading test data: %s", e)
            return []
    
    def get_model_response(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Get response from a model."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error getting response from model %s: %s", model, e)
            return f"Error: {str(e)}"
    
    def _initialize_retriever(self):
        """Initialize the retriever for RAG using LlamaIndex."""
        try:
            # Import necessary libraries for LlamaIndex
            import faiss

            from llama_index.vector_stores.faiss import FaissVectorStore
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.core.storage import StorageContext
            
            # Load FAISS index
            faiss_store_dir = Path("rag/faiss_store")
            if not faiss_store_dir.exists():
                logger.warning("FAISS index not found at %s", faiss_store_dir)
                return None
            
            try:
                # Check that all required files exist
                required_files = ["docstore.json", "index_store.json", "index_store.faiss"]
                missing_files = [f for f in required_files if not (faiss_store_dir / f).exists()]
                if missing_files:
                    logger.warning(f"Missing required files in FAISS store: {missing_files}")
                    return None
                
                # Initialize embedding model - must match the one used to create the index
                embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
                Settings.embed_model = embed_model
                
                # Load FAISS index
                faiss_index = faiss.read_index(str(faiss_store_dir / "index_store.faiss"))
                
                # Create vector store
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                
                # Initialize from disk directly without loading docstore
                # This avoids the UTF-8 decoding issues
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store
                )
                
                # Create the retriever with similarity search
                retriever = index.as_retriever(
                    similarity_top_k=3
                )
                
                logger.info("Successfully initialized RAG retriever using FAISS vector store")
                return retriever
                
            except Exception as e:
                logger.error(f"Error initializing LlamaIndex retriever: {e}")
                return None
            
        except ImportError as e:
            logger.error("Required LlamaIndex dependencies not installed: %s", e)
            return None
        except Exception as e:
            logger.error("Error initializing retriever: %s", e)
            return None
    
    def get_retrieved_context(self, query: str) -> str:
        """Get retrieved context for a query using LlamaIndex retriever."""
        if not self.retriever:
            logger.warning("Retriever not initialized, returning empty context")
            return ""
            
        try:
            # Retrieve nodes
            retrieval_results = self.retriever.retrieve(query)
            
            if not retrieval_results:
                logger.warning(f"No results found for query: {query}")
                return ""
                
            # Format context
            context_parts = []
            for i, result in enumerate(retrieval_results):
                try:
                    # In our FAISS-only implementation, node text might not be available
                    # Use similarity score and node_id for context
                    score = result.score if hasattr(result, "score") else "N/A"
                    node_id = result.node_id if hasattr(result, "node_id") else f"node_{i}"
                    
                    # Try to get text content, but handle gracefully if not available
                    text = ""
                    try:
                        if hasattr(result, "node") and hasattr(result.node, "get_content"):
                            text = result.node.get_content()
                    except Exception:
                        # If we can't get text content, just use a placeholder
                        text = f"[Medical document {i+1}]"
                    
                    # Add some metadata if available
                    metadata = {}
                    if hasattr(result, "node") and hasattr(result.node, "metadata"):
                        metadata = result.node.metadata
                    
                    source = metadata.get("source", "PubMed")
                    pmid = metadata.get("pmid", "")
                    title = metadata.get("title", "")
                    
                    # Add relevance info
                    relevance_info = f"Source: {source}"
                    if pmid:
                        relevance_info += f" (PMID: {pmid})"
                    relevance_info += f" (Score: {score:.4f})" if isinstance(score, float) else f" (Score: {score})"
                    
                    # Format the context piece
                    if text:
                        # Limit text length to avoid very long contexts
                        if len(text) > 800:
                            text = text[:800] + "..."
                        context_piece = f"[{i+1}] {text}\n{relevance_info}"
                    else:
                        # If no text is available, use the title and metadata
                        context_piece = f"[{i+1}] Title: {title}\n{relevance_info}"
                    
                    context_parts.append(context_piece)
                except Exception as e:
                    logger.warning(f"Error formatting retrieval result {i}: {e}")
                    context_parts.append(f"[{i+1}] [Content unavailable]\nScore: {getattr(result, 'score', 'N/A')}")
                
            context = "\n\n".join(context_parts)
            logger.info(f"Retrieved {len(retrieval_results)} context pieces for query: {query[:50]}...")
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "[Error retrieving RAG context]"
    
    def get_rag_response(self, query: str, model: str) -> str:
        """Get response using RAG with the specified model."""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return "Error: Retriever not initialized"
        
        try:
            # Get relevant context from the retriever
            context = self.get_retrieved_context(query)
            
            if not context:
                return "Error: No relevant context found"
            
            # Create a prompt with the retrieved context
            messages = [
                {"role": "system", "content": "You are a medical assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so."},
                {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
            ]
            
            # Get response from the specified model
            response = self.get_model_response(model=model, messages=messages)
            return response
        except Exception as e:
            logger.error("Error getting model response: %s", e)
            return f"Error: {str(e)}"
    
    def evaluate_response(
        self, 
        query: str, 
        ground_truth: str, 
        response: str
    ) -> Dict[str, float]:
        """Evaluate a single response using GPT-4."""
        try:
            # Factuality score (0-1)
            factuality_prompt = f"""
            Compare the ground truth and model response. Rate the factuality of the response on a scale of 0 to 1,
            where 1 means the response is completely factually consistent with the ground truth,
            and 0 means the response contains factual inaccuracies.
            
            Ground truth: {ground_truth}
            Response: {response}
            
            Return only a number between 0 and 1 representing your score. Do not include any explanation or other text.
            """
            
            factuality_response = self.client.chat.completions.create(
                model="gpt-4", # Using gpt-4 for evaluation
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of medical information. Respond with ONLY a number between 0 and 1 representing your score."},
                    {"role": "user", "content": factuality_prompt}
                ],
                temperature=0.1 # Low temperature for consistent scoring
            )
            try:
                # Try to extract a number from the response
                content = factuality_response.choices[0].message.content.strip()
                # First try direct conversion
                try:
                    factuality_score = float(content)
                except ValueError:
                    # If that fails, use regex to find a number
                    import re
                    match = re.search(r'\b(0|0\.\d+|1\.0|1)\b', content)
                    factuality_score = float(match.group(1)) if match else 0.5
                
                # Ensure score is in valid range
                factuality_score = min(max(factuality_score, 0.0), 1.0)
            except Exception as e:
                logger.error(f"Error parsing factuality score: {e}. Response: {factuality_response.choices[0].message.content}")
                factuality_score = 0.5  # Default score on error
            
            # Relevance score (0-1)
            relevance_prompt = f"""
            Rate the relevance of the response to the query on a scale of 0 to 1,
            where 1 means the response is completely relevant to the query,
            and 0 means the response is completely irrelevant.
            
            Query: {query}
            Response: {response}
            
            Return only a number between 0 and 1 representing your score. Do not include any explanation or other text.
            """
            
            relevance_response = self.client.chat.completions.create(
                model="gpt-4", # Using gpt-4 for evaluation
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of medical information. Respond with ONLY a number between 0 and 1 representing your score."},
                    {"role": "user", "content": relevance_prompt}
                ],
                temperature=0.1 # Low temperature for consistent scoring
            )
            try:
                # Try to extract a number from the response
                content = relevance_response.choices[0].message.content.strip()
                # First try direct conversion
                try:
                    relevance_score = float(content)
                except ValueError:
                    # If that fails, use regex to find a number
                    import re
                    match = re.search(r'\b(0|0\.\d+|1\.0|1)\b', content)
                    relevance_score = float(match.group(1)) if match else 0.5
                
                # Ensure score is in valid range
                relevance_score = min(max(relevance_score, 0.0), 1.0)
            except Exception as e:
                logger.error(f"Error parsing relevance score: {e}. Response: {relevance_response.choices[0].message.content}")
                relevance_score = 0.5  # Default score on error
            
            return {
                "factuality": factuality_score,
                "relevance": relevance_score,
                "average": (factuality_score + relevance_score) / 2
            }
        except Exception as e:
            logger.error("Error evaluating response: %s", e)
            return {"factuality": 0.0, "relevance": 0.0, "average": 0.0}

    def log_to_langfuse(
        self,
        example_id: str,
        query: str,
        ground_truth: str,
        responses: Dict[str, str],
        scores: Dict[str, Dict[str, float]]
    ) -> Optional[str]:
        """Log evaluation results to Langfuse."""
        if not self.langfuse_enabled or not self.langfuse:
            logger.warning("Langfuse logging disabled or not initialized")
            return None
            
        try:
            # Use a consistent trace name for all examples in this evaluation run
            if not hasattr(self, 'run_trace_id'):
                self.run_trace_id = f"medical-qa-eval-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"

            # Create a unique trace ID for this example
            trace_id = f"{self.run_trace_id}-example-{example_id}"
            
            # Add all relevant metadata
            metadata = {
                "example_id": example_id,
                "query": query,
                "ground_truth": ground_truth,
                "baseline_model": self.baseline_model,
                "fine_tuned_model": self.fine_tuned_model or "N/A",
                "use_rag": str(self.use_rag),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Add score summaries directly to metadata
            for model_type, model_scores in scores.items():
                if model_scores:
                    for metric, value in model_scores.items():
                        try:
                            metadata[f"{model_type}_{metric}"] = float(value)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert score {metric}='{value}' to float. Storing as string.")
                            metadata[f"{model_type}_{metric}"] = str(value)
            
            # Create a trace for this example - use a consistent name for all examples
            trace = self.langfuse.trace(
                name="Medical QA Evaluation",
                id=trace_id,
                metadata=metadata,
                tags=["evaluation", "medical-qa"]
            )
            
            # Log the user query
            trace.generation(
                name="User Query",
                model="user",
                input=query,
                output=query
            )
            
            # Log the ground truth directly as a generation
            trace.generation(
                name="Ground Truth",
                model="ground_truth",
                input=query,
                output=ground_truth
            )
            
            # Log each model response
            for model_type, response in responses.items():
                if not response or response.strip() == "":
                    logger.warning(f"Empty response for {model_type} in example {example_id}, skipping")
                    continue
                    
                model_scores = scores.get(model_type, {})
                
                # Determine display name and actual model name
                display_name = ""
                actual_model_name = ""

                if model_type == "baseline":
                    display_name = f"Baseline ({self.baseline_model})"
                    actual_model_name = self.baseline_model
                elif model_type == "fine_tuned":
                    display_name = f"Fine-tuned ({self.fine_tuned_model})"
                    actual_model_name = self.fine_tuned_model
                elif model_type == "baseline_rag":
                    display_name = f"Baseline + RAG ({self.baseline_model})"
                    actual_model_name = self.baseline_model
                elif model_type == "fine_tuned_rag":
                    display_name = f"Fine-tuned + RAG ({self.fine_tuned_model})"
                    actual_model_name = self.fine_tuned_model
                else:
                    display_name = model_type
                    actual_model_name = model_type

                # Ensure model name is not empty
                if not actual_model_name:
                    actual_model_name = "unknown_model"

                # Log the model response
                gen = trace.generation(
                    name=display_name,
                    model=actual_model_name,
                    input=query,
                    output=response,
                    metadata=model_scores
                )
                
                # Add scores directly to the generation
                for metric, value in model_scores.items():
                    try:
                        float_value = min(max(float(value), 0.0), 1.0)
                        gen.score(name=metric, value=float_value)
                    except Exception as score_error:
                        logger.error(f"Error logging score {metric}={value} to Langfuse: {score_error}")
            
            # Flush to ensure data is sent
            self.langfuse.flush()
            logger.info(f"Successfully logged evaluation example {example_id} to Langfuse")
            return trace_id
            
        except Exception as e:
            logger.error(f"Error logging to Langfuse: {e}")
            return None
    
    def run_evaluation(self) -> Dict[str, Dict[str, List[float]]]:
        """Run evaluation on test data."""
        # Load test data
        test_examples = self.load_test_data()
        if not test_examples:
            return {}
        
        # Initialize results dictionary
        results = {
            "baseline": {"factuality": [], "relevance": [], "average": []},
            "baseline_rag": {"factuality": [], "relevance": [], "average": []},
            "fine_tuned": {"factuality": [], "relevance": [], "average": []},
            "fine_tuned_rag": {"factuality": [], "relevance": [], "average": []}
        }
        
        # Evaluate each example
        for i, example in enumerate(tqdm(test_examples, desc="Evaluating")):
            # Extract query and ground truth
            messages = example["messages"]
            query = ""
            ground_truth = ""
            
            # Find user and assistant messages
            for msg in messages:
                if msg["role"] == "user":
                    query = msg["content"]
                elif msg["role"] == "assistant":
                    ground_truth = msg["content"]
            
            if not query or not ground_truth:
                logger.warning(f"Skipping example {i}: missing query or ground truth")
                continue
            
            # Get responses for each configuration
            responses = {}
            scores = {}
        
            # 1. Baseline model without RAG
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": query}
                ]
                baseline_response = self.get_model_response(
                    model=self.baseline_model,
                    messages=messages
                )
                responses["baseline"] = baseline_response
                
                # Evaluate
                baseline_scores = self.evaluate_response(
                    query=query,
                    ground_truth=ground_truth,
                    response=baseline_response
                )
                scores["baseline"] = baseline_scores
                
                # Store scores
                for metric, value in baseline_scores.items():
                    results["baseline"][metric].append(value)
            except Exception as e:
                logger.error(f"Error evaluating baseline model: {e}")
            
            # 2. Fine-tuned model without RAG (if available)
            if self.fine_tuned_model:
                try:
                    messages = [
                        {"role": "system", "content": "You are a helpful medical assistant."},
                        {"role": "user", "content": query}
                    ]
                    fine_tuned_response = self.get_model_response(
                        model=self.fine_tuned_model,
                        messages=messages
                    )
                    responses["fine_tuned"] = fine_tuned_response
                    
                    # Evaluate
                    fine_tuned_scores = self.evaluate_response(
                        query=query,
                        ground_truth=ground_truth,
                        response=fine_tuned_response
                    )
                    scores["fine_tuned"] = fine_tuned_scores
                    
                    # Store scores
                    for metric, value in fine_tuned_scores.items():
                        results["fine_tuned"][metric].append(value)
                except Exception as e:
                    logger.error(f"Error evaluating fine-tuned model: {e}")
            
            # 3. Models with RAG (if enabled)
            if self.use_rag and self.retriever:
                # Baseline model with RAG
                try:
                    baseline_rag_response = self.get_rag_response(query, self.baseline_model)
                    responses["baseline_rag"] = baseline_rag_response
                    
                    # Evaluate
                    baseline_rag_scores = self.evaluate_response(
                        query=query,
                        ground_truth=ground_truth,
                        response=baseline_rag_response
                    )
                    scores["baseline_rag"] = baseline_rag_scores
                    
                    # Store scores
                    for metric, value in baseline_rag_scores.items():
                        results["baseline_rag"][metric].append(value)
                except Exception as e:
                    logger.error(f"Error evaluating baseline model with RAG: {e}")
                
                # Fine-tuned model with RAG (if available)
                if self.fine_tuned_model:
                    try:
                        fine_tuned_rag_response = self.get_rag_response(query, self.fine_tuned_model)
                        responses["fine_tuned_rag"] = fine_tuned_rag_response
                        
                        # Evaluate
                        fine_tuned_rag_scores = self.evaluate_response(
                            query=query,
                            ground_truth=ground_truth,
                            response=fine_tuned_rag_response
                        )
                        scores["fine_tuned_rag"] = fine_tuned_rag_scores
                        
                        # Store scores
                        for metric, value in fine_tuned_rag_scores.items():
                            results["fine_tuned_rag"][metric].append(value)
                    except Exception as e:
                        logger.error(f"Error evaluating fine-tuned model with RAG: {e}")
            
            # Log to Langfuse
            if self.langfuse_enabled and self.langfuse:
                trace_id = self.log_to_langfuse(
                    example_id=f"example_{i}",
                    query=query,
                    ground_truth=ground_truth,
                    responses=responses,
                    scores=scores
                )
                if trace_id:
                    logger.info(f"Logged example {i} to Langfuse with trace ID: {trace_id}")
        
        # Calculate summary statistics
        summary = {}
        for model_type, metrics in results.items():
            summary[model_type] = {}
            for metric, values in metrics.items():
                if values:
                    summary[model_type][f"{metric}_mean"] = float(np.mean(values))
                    summary[model_type][f"{metric}_std"] = float(np.std(values))
                else:
                    summary[model_type][f"{metric}_mean"] = 0.0
                    summary[model_type][f"{metric}_std"] = 0.0
        
        # Save results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_file = RESULTS_DIR / "evaluation_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "detailed": results
            }, f, indent=2, ensure_ascii=False)
        
        # Generate report
        self.generate_report(summary, results)
        
        logger.info(f"Evaluation completed. Results saved to {results_file}")
        return summary

    def generate_report(self, summary: Dict, results: Dict) -> None:
        """Generate a markdown report of the evaluation results."""
        report_path = RESULTS_DIR / "report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Model Evaluation Report\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            
            # Create a table of average scores
            f.write("### Average Scores\n\n")
            f.write("| Model | Factuality (Mean ± SD) | Relevance (Mean ± SD) | Average (Mean ± SD) |\n")
            f.write("|-------|----------------------|----------------------|---------------------|\n")
            
            for model, scores in summary.items():
                fact_str = f"{scores.get('factuality_mean', 0.0):.3f} ± {scores.get('factuality_std', 0.0):.3f}"
                rel_str = f"{scores.get('relevance_mean', 0.0):.3f} ± {scores.get('relevance_std', 0.0):.3f}"
                avg_str = f"{scores.get('average_mean', 0.0):.3f} ± {scores.get('average_std', 0.0):.3f}"
                f.write(f"| {model.replace('_', ' ').title()} | {fact_str} | {rel_str} | {avg_str} |\n")
            
            # Configuration
            f.write("\n## Configuration\n\n")
            f.write(f"- **Baseline Model:** {self.baseline_model}\n")
            if self.fine_tuned_model:
                f.write(f"- **Fine-tuned Model:** {self.fine_tuned_model}\n")
            f.write(f"- **RAG Enabled:** {self.use_rag}\n")
            f.write(f"- **Evaluation Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Detailed Results
            f.write("\n## Detailed Results\n\n")
            f.write("The detailed evaluation results are available in JSON format at `evaluation_results.json`.\n")
        
        logger.info(f"Report generated: {report_path}")

def main() -> None:
    """Run the evaluation harness with command line arguments."""
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument(
        "--baseline",
        type=str,
        default=BASELINE_MODEL,
        help="Baseline model name"
    )
    parser.add_argument(
        "--fine-tuned",
        type=str,
        default=None,
        help="Fine-tuned model name"
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable RAG for evaluation"
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_false",
        dest="langfuse",
        help="Disable Langfuse logging"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to evaluate"
    )
    
    args = parser.parse_args()
    
    # Initialize and run evaluation
    try:
        harness = EvaluationHarness(
            baseline_model=args.baseline,
            fine_tuned_model=args.fine_tuned,
            use_rag=args.use_rag,
            langfuse_enabled=args.langfuse,
            limit=args.limit
        )
        
        results = harness.run_evaluation()
        print("\nEvaluation Results:")
        print(json.dumps(results, indent=2))
        return 0
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())