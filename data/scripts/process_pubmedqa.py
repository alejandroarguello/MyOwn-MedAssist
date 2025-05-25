#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process PubMedQA dataset:
1. Clean and deduplicate data
2. Convert to JSONL format suitable for OpenAI fine-tuning
3. Ensure no PHI is present
"""

import os
import json
import jsonlines
import pandas as pd
from pathlib import Path
import logging
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = Path("../raw/pubmedqa")
PROCESSED_DATA_DIR = Path("../processed")
OUTPUT_FILE = PROCESSED_DATA_DIR / "pubmedqa_processed.jsonl"

def load_pubmedqa_data():
    """Load PubMedQA dataset from extracted files."""
    logger.info("Loading PubMedQA data")
    
    # Find the main dataset file
    main_data_path = list(RAW_DATA_DIR.glob("**/ori_pqal.json"))
    test_ground_truth_path = list(RAW_DATA_DIR.glob("**/test_ground_truth.json"))
    
    if not main_data_path:
        raise FileNotFoundError("Could not find ori_pqal.json in the PubMedQA directory")
    
    main_data_path = main_data_path[0]
    logger.info(f"Found main data file at {main_data_path}")
    
    # Load the main dataset
    with open(main_data_path, 'r') as f:
        all_data = json.load(f)
    
    logger.info(f"Loaded {len(all_data)} examples from PubMedQA dataset")
    
    # Split into train/dev/test (80/10/10 split)
    # In a real scenario, we'd use the official splits if available
    all_examples = list(all_data.items())
    total = len(all_examples)
    
    train_split = int(0.8 * total)
    dev_split = int(0.9 * total)
    
    train_data = dict(all_examples[:train_split])
    dev_data = dict(all_examples[train_split:dev_split])
    test_data = dict(all_examples[dev_split:])
    
    logger.info(f"Split into {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test examples")
    
    return train_data, dev_data, test_data

def clean_text(text):
    """Clean text by removing excessive whitespace, etc."""
    if not text:
        return ""
    
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def format_for_openai(example_id, example_data):
    """Format a PubMedQA example for OpenAI fine-tuning."""
    
    # Extract relevant fields from the correct structure
    question = clean_text(example_data.get('QUESTION', ''))
    
    # Contexts is an array in the original data, join them with newlines
    contexts = example_data.get('CONTEXTS', [])
    context = clean_text('\n'.join(contexts) if contexts else '')
    
    # Get the long answer if available
    long_answer = clean_text(example_data.get('LONG_ANSWER', ''))
    
    # Check if this is a yes/no/maybe question
    final_decision = example_data.get('final_decision', '')
    
    # Only use examples that have both question and context
    if not question or not context:
        logger.warning(f"Skipping example {example_id} due to missing question or context")
        return None
    
    # Combine context and question for the prompt
    prompt = f"Context: {context}\n\nQuestion: {question}"
    
    # For the completion, use the long answer if available, otherwise use the yes/no/maybe answer
    completion = long_answer if long_answer else final_decision
    
    # Skip examples with empty completions
    if not completion:
        logger.warning(f"Skipping example {example_id} due to missing answer")
        return None
    
    # Format for OpenAI Chat fine-tuning
    formatted = {
        "messages": [
            {"role": "system", "content": "You are a medical assistant that provides accurate information based on medical literature."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
    }
    
    return formatted

def process_pubmedqa():
    """Process PubMedQA dataset and convert to JSONL for fine-tuning."""
    logger.info("Processing PubMedQA dataset")
    
    # Load data
    train_data, dev_data, test_data = load_pubmedqa_data()
    
    # Combine all data for processing
    all_data = {}
    for data_dict in [train_data, dev_data, test_data]:
        all_data.update(data_dict)
    
    logger.info(f"Total examples: {len(all_data)}")
    
    # Process each example with the correct format
    processed_examples = []
    for example_id, example_data in all_data.items():
        formatted = format_for_openai(example_id, example_data)
        if formatted:  # Only add if not None (skipped examples return None)
            processed_examples.append(formatted)
    
    logger.info(f"After processing: {len(processed_examples)} valid examples")
    
    # Save to JSONL file
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        for example in processed_examples:
            writer.write(example)
    
    logger.info(f"Saved {len(processed_examples)} examples to {OUTPUT_FILE}")
    
    return len(processed_examples)

def main():
    """Main function to process PubMedQA dataset."""
    logger.info("Starting PubMedQA processing")
    
    num_examples = process_pubmedqa()
    
    logger.info(f"Processing complete. {num_examples} examples processed.")

if __name__ == "__main__":
    main()
