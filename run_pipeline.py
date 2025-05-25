#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline runner for MyOwn-MedAssist:
1. Downloads and processes datasets
2. Curates Q-A pairs
3. Fine-tunes model
4. Builds vector store
5. Runs evaluation
"""

import os
import argparse
import subprocess
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.absolute()

def run_command(command, cwd=None):
    """Run a shell command and log output."""
    if cwd is None:
        cwd = PROJECT_ROOT
    
    logger.info(f"Running command: {command}")
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        if return_code != 0:
            error = process.stderr.read()
            logger.error(f"Command failed with return code {return_code}")
            logger.error(error)
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False

def download_datasets():
    """Download datasets."""
    logger.info("Step 1: Downloading datasets")
    
    data_scripts_dir = PROJECT_ROOT / "data" / "scripts"
    return run_command("python download_datasets.py", cwd=data_scripts_dir)

def process_datasets():
    """Process datasets."""
    logger.info("Step 2: Processing datasets")
    
    data_scripts_dir = PROJECT_ROOT / "data" / "scripts"
    
    # Process each dataset
    success = True
    success = success and run_command("python process_pubmedqa.py", cwd=data_scripts_dir)
    success = success and run_command("python process_medmcqa.py", cwd=data_scripts_dir)
    success = success and run_command("python process_synthea.py", cwd=data_scripts_dir)
    success = success and run_command("python curate_synthea_data.py", cwd=data_scripts_dir)
    
    return success

def curate_qa_pairs():
    """Curate Q-A pairs."""
    logger.info("Step 4: Curating Q-A pairs")
    
    fine_tuning_dir = PROJECT_ROOT / "fine_tuning"
    return run_command("python3 curate_qa_pairs.py", cwd=fine_tuning_dir)

def fine_tune_model(use_full_set=False):
    """Fine-tune model.
    
    Args:
        use_full_set: If True, use the full training set instead of the small one
    """
    logger.info("Step 5: Fine-tuning model")
    
    fine_tuning_dir = PROJECT_ROOT / "fine_tuning"
    command = "python openai_finetune.py"
    if use_full_set:
        command += " --full"
    
    return run_command(command, cwd=fine_tuning_dir)

def build_vector_store():
    """Build vector store."""
    logger.info("Step 6: Building vector store")
    
    rag_dir = PROJECT_ROOT / "rag"
    return run_command("python build_vector_store.py", cwd=rag_dir)

def analyze_data():
    """Analyze datasets and generate visualizations."""
    logger.info("Step 3: Analyzing datasets")
    
    data_analysis_dir = PROJECT_ROOT / "data_analysis" / "scripts"
    return run_command("python analyze_datasets.py", cwd=data_analysis_dir)

def run_evaluation(fine_tuned_model=None, use_rag=False):
    """Run evaluation."""
    logger.info("Step 7: Running evaluation")
    
    evaluation_dir = PROJECT_ROOT / "evaluation"
    
    command = "python harness.py"
    if fine_tuned_model:
        command += f" --fine-tuned {fine_tuned_model}"
    if use_rag:
        command += " --rag"
    
    return run_command(command, cwd=evaluation_dir)

def start_api():
    """Start API server."""
    logger.info("Starting API server")
    
    api_dir = PROJECT_ROOT / "api"
    return run_command("uvicorn main:app --host 0.0.0.0 --port 8000 --reload", cwd=api_dir)

def start_frontend():
    """Start frontend."""
    logger.info("Starting frontend")
    
    frontend_dir = PROJECT_ROOT / "frontend"
    return run_command("streamlit run app.py", cwd=frontend_dir)

def main():
    """Main function to run pipeline."""
    parser = argparse.ArgumentParser(description="Run MyOwn-MedAssist pipeline")
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--process", action="store_true", help="Process datasets")
    parser.add_argument("--curate", action="store_true", help="Curate Q-A pairs")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune model (uses small training set by default)")
    parser.add_argument("--full-training", action="store_true", help="Use full training set for fine-tuning (larger, more comprehensive)")
    parser.add_argument("--vectorstore", action="store_true", help="Build vector store")
    parser.add_argument("--analyze", action="store_true", help="Analyze datasets and generate visualizations")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--model", type=str, help="Fine-tuned model ID for evaluation")
    parser.add_argument("--rag", action="store_true", help="Use RAG for evaluation")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--frontend", action="store_true", help="Start frontend")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()
    
    # Check if any argument is provided
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run all steps if --all is provided
    if args.all:
        args.download = True
        args.process = True
        args.curate = True
        args.finetune = True
        args.vectorstore = True
        args.analyze = True
        args.evaluate = True
    
    # Run steps
    if args.download:
        if not download_datasets():
            logger.error("Dataset download failed")
            return
    
    if args.process:
        if not process_datasets():
            logger.error("Dataset processing failed")
            return
    
    if args.analyze:
        if not analyze_data():
            logger.error("Dataset analysis failed")
            return
    
    if args.curate:
        if not curate_qa_pairs():
            logger.error("Q-A pair curation failed")
            return
    
    if args.finetune:
        if not fine_tune_model(use_full_set=args.full_training):
            logger.error("Fine-tuning failed")
            return
    
    if args.vectorstore:
        if not build_vector_store():
            logger.error("Vector store building failed")
            return
    
    if args.evaluate:
        if not run_evaluation(args.model, args.rag):
            logger.error("Evaluation failed")
            return
    
    if args.api:
        start_api()
    
    if args.frontend:
        start_frontend()

if __name__ == "__main__":
    main()
