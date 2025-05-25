#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune OpenAI's gpt-3.5-turbo model using curated Q-A pairs.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from openai import OpenAI
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FINETUNING_DATA_DIR = Path("./data")
TRAIN_FILE = FINETUNING_DATA_DIR / "train.jsonl"
SMALL_TRAIN_FILE = FINETUNING_DATA_DIR / "train_small.jsonl"
OUTPUT_DIR = Path("./models")
MODEL_NAME = "gpt-3.5-turbo"  # Base model to fine-tune

def validate_api_key():
    """Validate that the OpenAI API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set. Please set it in a .env file or export it.")
        return False
    return True

def upload_training_file(client, file_path):
    """Upload training file to OpenAI."""
    logger.info(f"Uploading training file: {file_path}")
    
    # Check if file exists
    if not Path(file_path).exists():
        logger.error(f"Training file not found: {file_path}")
        logger.error("Please run curate_qa_pairs.py first.")
        return None
    
    with open(file_path, "rb") as file:
        response = client.files.create(
            file=file,
            purpose="fine-tune"
        )
    
    file_id = response.id
    logger.info(f"File uploaded with ID: {file_id}")
    
    return file_id

def create_fine_tuning_job(client, file_id, model=MODEL_NAME):
    """Create a fine-tuning job."""
    logger.info(f"Creating fine-tuning job for file {file_id} using model {model}")
    
    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        suffix="myown-medassist"  # Custom suffix for the fine-tuned model
    )
    
    job_id = response.id
    logger.info(f"Fine-tuning job created with ID: {job_id}")
    
    return job_id

def estimate_cost(training_file, is_small=False):
    """Estimate the cost of fine-tuning based on file size and token count."""
    try:
        # Count lines in the JSONL file to get example count
        with open(training_file, 'r') as f:
            example_count = sum(1 for _ in f)
        
        # Estimate token count (rough approximation)
        # Average tokens per example based on our dataset analysis
        avg_tokens_per_example = 500  # Conservative estimate
        estimated_tokens = example_count * avg_tokens_per_example
        
        # Current OpenAI pricing (as of May 2025)
        # Training: $0.008 per 1K tokens
        training_cost = (estimated_tokens / 1000) * 0.008
        
        # Estimate training time
        estimated_hours = 1 if is_small else 3  # Rough estimate
        
        logger.info(f"\nEstimated Fine-tuning Cost and Time:")
        logger.info(f"Training set: {'Small' if is_small else 'Full'}")
        logger.info(f"Examples: {example_count}")
        logger.info(f"Estimated tokens: {estimated_tokens:,}")
        logger.info(f"Estimated cost: ${training_cost:.2f}")
        logger.info(f"Estimated training time: {estimated_hours} hour{'s' if estimated_hours > 1 else ''}")
        logger.info(f"Note: These are estimates. Actual costs may vary.")
        
        return training_cost
    except Exception as e:
        logger.error(f"Error estimating cost: {e}")
        return None

def monitor_fine_tuning_job(client, job_id):
    """Monitor the fine-tuning job until it completes and log metrics to W&B."""
    logger.info(f"Monitoring fine-tuning job: {job_id}")
    
    # Initialize W&B run
    wandb.init(
        project="medassist-openai-finetuning",
        name=f"finetune-{MODEL_NAME}-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "base_model": MODEL_NAME,
            "job_id": job_id,
            "training_set_size": "small" if not args.full else "full"
        }
    )
    
    # Track metrics
    metrics_history = []
    
    while True:
        # Get the status of the fine-tuning job
        response = client.fine_tuning.jobs.retrieve(job_id)
        status = response.status
        
        logger.info(f"Job status: {status}")
        wandb.log({"status": status})
        
        # Log training metrics if available
        if hasattr(response, 'training_metrics') and response.training_metrics:
            metrics = response.training_metrics
            metrics_history.append(metrics)
            
            # Log metrics to W&B
            if hasattr(metrics, 'train_loss'):
                wandb.log({"train_loss": metrics.train_loss})
            if hasattr(metrics, 'train_accuracy'):
                wandb.log({"train_accuracy": metrics.train_accuracy})
            if hasattr(metrics, 'train_token_accuracy'):
                wandb.log({"train_token_accuracy": metrics.train_token_accuracy})
        
        # Log events if available
        if hasattr(response, 'events') and response.events.data:
            for event in response.events.data:
                wandb.log({"event": event.message, "event_level": event.level})
                logger.info(f"Event: {event.message} (Level: {event.level})")
        
        if status in ["succeeded", "failed", "cancelled"]:
            break
            
        # Wait before checking again
        time.sleep(60)  # Check every minute
    
    if status == "succeeded":
        fine_tuned_model = response.fine_tuned_model
        
        # Save model info
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        model_info = {
            "model_id": fine_tuned_model,
            "base_model": MODEL_NAME,
            "training_file": str(TRAIN_FILE),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "job_id": job_id,
            "wandb_run_id": wandb.run.id
        }
        
        with open(OUTPUT_DIR / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model info saved to {OUTPUT_DIR / 'model_info.json'}")
        
        # Log final status
        wandb.log({
            "final_status": status,
        })
        
        # Log the fine_tuned_model as a config value, not as a metric
        wandb.config.update({"fine_tuned_model": fine_tuned_model if fine_tuned_model else ""}, allow_val_change=True)

        # Fetch and log the full curve
        result_file_id = response.result_files[0].id
        csv_bytes = client.files.retrieve_content(result_file_id)
        df = pd.read_csv(io.BytesIO(csv_bytes))

        # Log the curve
        wandb.log({"training_curve":
        wandb.plot.line_series(xs=df["step"],
                                ys=[df["train_loss"]],
                                keys=["train_loss"],
                                title="Fine-tuning loss",
                                xname="Step")})
        
        wandb.finish()
        
        return fine_tuned_model
    else:
        logger.error(f"Fine-tuning failed with status: {status}")
        
        # Log failure to W&B
        wandb.log({"final_status": "failed", "failure_reason": status})
        wandb.finish()
        
        return None

def main():
    """Main function to run OpenAI fine-tuning."""
    logger.info("Starting OpenAI fine-tuning")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune OpenAI's gpt-3.5-turbo model")
    parser.add_argument("--full", action="store_true", help="Use full training set (larger, more comprehensive)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--estimate-cost", action="store_true", help="Estimate cost without starting fine-tuning")
    args = parser.parse_args()
    
    # Check for W&B API key in environment variables
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        logger.warning("WANDB_API_KEY not found in .env file. W&B logging may not work properly.")
        logger.warning("Add WANDB_API_KEY=your_api_key to your .env file for W&B integration.")
    
    # Disable W&B if requested
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("Weights & Biases logging disabled")
    
    # Validate API key
    if not validate_api_key():
        return
    
    # Determine which training file to use
    training_file = TRAIN_FILE if args.full else SMALL_TRAIN_FILE
    training_set_type = "full" if args.full else "small"
    
    # Check if training file exists
    if not training_file.exists():
        logger.error(f"{training_set_type.capitalize()} training file not found: {training_file}")
        logger.error("Please run curate_qa_pairs.py first.")
        return
    
    # Calculate and display estimated cost
    estimate_cost(training_file, not args.full)
    
    # If only estimating cost, exit
    if args.estimate_cost:
        logger.info("Cost estimation complete. Exiting without starting fine-tuning.")
        return
    
    # Confirm with user
    logger.info(f"Ready to start fine-tuning with {training_set_type} training set.")
    logger.info(f"Training file: {training_file}")
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Upload training file
    file_id = upload_training_file(client, training_file)
    if not file_id:
        return
    
    # Create fine-tuning job
    job_id = create_fine_tuning_job(client, file_id)
    
    # Monitor fine-tuning job
    fine_tuned_model = monitor_fine_tuning_job(client, job_id)
    
    if fine_tuned_model:
        logger.info(f"Fine-tuning complete. Model: {fine_tuned_model}")
    else:
        logger.error("Fine-tuning failed.")

if __name__ == "__main__":
    main()
