#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Curate Q-A pairs for fine-tuning:
1. Combine processed data from all sources
2. Select high-quality examples
3. Split into train and test sets
4. Format for OpenAI fine-tuning
"""

import os
import json
import jsonlines
import pandas as pd
from pathlib import Path
import logging
import random
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DATA_DIR = Path("../data/processed")
OUTPUT_DIR = Path("./data")
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
SMALL_TRAIN_FILE = OUTPUT_DIR / "train_small.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"
MAX_TRAIN_EXAMPLES = 2000  # Limit training examples to keep fine-tuning manageable
SMALL_TRAIN_EXAMPLES = 500  # Smaller training set for faster fine-tuning
TEST_SIZE = 0.2  # 20% of data for testing

def load_processed_data(small_set=False):
    """Load all processed datasets.
    
    Args:
        small_set: If True, create a smaller balanced dataset
        
    Returns:
        List of balanced examples
    """
    logger.info("Loading processed datasets")
    
    pubmedqa_examples = []
    medmcqa_examples = []
    synthea_examples = []
    
    # Load PubMedQA
    pubmedqa_file = PROCESSED_DATA_DIR / "pubmedqa_processed.jsonl"
    if pubmedqa_file.exists():
        with jsonlines.open(pubmedqa_file) as reader:
            pubmedqa_examples = list(reader)
        logger.info(f"Loaded {len(pubmedqa_examples)} PubMedQA examples")
    
    # Load MedMCQA
    medmcqa_file = PROCESSED_DATA_DIR / "medmcqa_processed.jsonl"
    if medmcqa_file.exists():
        with jsonlines.open(medmcqa_file) as reader:
            medmcqa_examples = list(reader)
        logger.info(f"Loaded {len(medmcqa_examples)} MedMCQA examples")
    
    # Load Synthea
    synthea_file = PROCESSED_DATA_DIR / "synthea_curated.jsonl"  # Using curated version
    if synthea_file.exists():
        with jsonlines.open(synthea_file) as reader:
            synthea_examples = list(reader)
        logger.info(f"Loaded {len(synthea_examples)} Synthea examples")
    
    # Balance datasets based on analysis insights
    balanced_examples = balance_datasets(pubmedqa_examples, medmcqa_examples, synthea_examples, small_set=small_set)
    
    logger.info(f"Total examples in {'small' if small_set else 'full'} balanced dataset: {len(balanced_examples)}")
    return balanced_examples

def balance_datasets(pubmedqa_examples, medmcqa_examples, synthea_examples, small_set=False):
    """Balance datasets to create a representative fine-tuning dataset.
    
    Args:
        pubmedqa_examples: List of PubMedQA examples
        medmcqa_examples: List of MedMCQA examples
        synthea_examples: List of Synthea examples
        small_set: If True, create a smaller balanced dataset
        
    Returns:
        List of balanced examples
    """
    if small_set:
        logger.info("Balancing datasets for small fine-tuning set")
        # Target counts for small training set (proportionally reduced)
        pubmedqa_target = min(175, len(pubmedqa_examples))  # 25% of full set
        medmcqa_target = min(175, len(medmcqa_examples))    # 25% of full set
        synthea_target = min(75, len(synthea_examples))     # 25% of full set
    else:
        logger.info("Balancing datasets for full fine-tuning set")
        # Target counts based on analysis and recommendations
        pubmedqa_target = min(700, len(pubmedqa_examples))
        medmcqa_target = min(700, len(medmcqa_examples))
        synthea_target = min(300, len(synthea_examples))
    
    # Sample from each dataset
    pubmedqa_sample = random.sample(pubmedqa_examples, pubmedqa_target) if len(pubmedqa_examples) > pubmedqa_target else pubmedqa_examples
    medmcqa_sample = random.sample(medmcqa_examples, medmcqa_target) if len(medmcqa_examples) > medmcqa_target else medmcqa_examples
    synthea_sample = random.sample(synthea_examples, synthea_target) if len(synthea_examples) > synthea_target else synthea_examples
    
    # Log dataset composition
    total_examples = len(pubmedqa_sample) + len(medmcqa_sample) + len(synthea_sample)
    logger.info(f"Dataset composition ({'small' if small_set else 'full'} set):")
    logger.info(f"  - PubMedQA: {len(pubmedqa_sample)} examples ({len(pubmedqa_sample)/total_examples*100:.1f}%)")
    logger.info(f"  - MedMCQA: {len(medmcqa_sample)} examples ({len(medmcqa_sample)/total_examples*100:.1f}%)")
    logger.info(f"  - Synthea: {len(synthea_sample)} examples ({len(synthea_sample)/total_examples*100:.1f}%)")
    
    # Combine samples
    balanced_examples = []
    balanced_examples.extend(pubmedqa_sample)
    balanced_examples.extend(medmcqa_sample)
    balanced_examples.extend(synthea_sample)
    
    # Shuffle to ensure random distribution
    random.shuffle(balanced_examples)
    
    return balanced_examples

def filter_examples(examples):
    """Filter examples to select high-quality ones."""
    logger.info("Filtering examples for quality")
    
    filtered_examples = []
    
    for example in examples:
        # Check if example has valid messages
        if "messages" not in example or not isinstance(example["messages"], list):
            continue
        
        # Check if messages have required roles and content
        valid_messages = True
        for msg in example["messages"]:
            if "role" not in msg or "content" not in msg:
                valid_messages = False
                break
            
            # Check for empty content
            if not msg["content"].strip():
                valid_messages = False
                break
        
        if not valid_messages:
            continue
        
        # Check if there's a user question and assistant response
        has_user = False
        has_assistant = False
        for msg in example["messages"]:
            if msg["role"] == "user":
                has_user = True
            elif msg["role"] == "assistant":
                has_assistant = True
        
        if not (has_user and has_assistant):
            continue
        
        # Add example to filtered list
        filtered_examples.append(example)
    
    logger.info(f"After filtering: {len(filtered_examples)} examples")
    
    return filtered_examples

def sample_examples(examples, max_examples):
    """Sample a limited number of examples if needed."""
    if len(examples) <= max_examples:
        return examples
    
    logger.info(f"Sampling {max_examples} examples from {len(examples)}")
    
    # Shuffle and sample
    random.seed(42)  # For reproducibility
    random.shuffle(examples)
    
    return examples[:max_examples]

def split_train_test(examples, test_size=TEST_SIZE):
    """Split examples into train and test sets."""
    logger.info(f"Splitting into train and test sets (test_size={test_size})")
    
    train_examples, test_examples = train_test_split(
        examples, test_size=test_size, random_state=42
    )
    
    logger.info(f"Train set: {len(train_examples)} examples")
    logger.info(f"Test set: {len(test_examples)} examples")
    
    return train_examples, test_examples

def save_jsonl(examples, output_file):
    """Save examples to JSONL file."""
    os.makedirs(output_file.parent, exist_ok=True)
    
    with jsonlines.open(output_file, mode='w') as writer:
        for example in examples:
            writer.write(example)
    
    logger.info(f"Saved {len(examples)} examples to {output_file}")

def main():
    """Main function to curate Q-A pairs."""
    logger.info("Starting Q-A pair curation")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process full training set
    logger.info("Processing full training set")
    all_examples = load_processed_data(small_set=False)
    
    if not all_examples:
        logger.warning("No examples found. Please run data processing scripts first.")
        return
    
    # Filter for quality
    filtered_examples = filter_examples(all_examples)
    
    # Sample if needed
    sampled_examples = sample_examples(filtered_examples, MAX_TRAIN_EXAMPLES)
    
    # Split into train and test
    train_examples, test_examples = split_train_test(sampled_examples)
    
    # Save full training set
    save_jsonl(train_examples, TRAIN_FILE)
    save_jsonl(test_examples, TEST_FILE)
    
    # Process small training set
    logger.info("\nProcessing small training set")
    small_examples = load_processed_data(small_set=True)
    
    # Filter for quality
    small_filtered = filter_examples(small_examples)
    
    # Sample if needed
    small_sampled = sample_examples(small_filtered, SMALL_TRAIN_EXAMPLES)
    
    # Save small training set (using same test set)
    save_jsonl(small_sampled, SMALL_TRAIN_FILE)
    
    logger.info("Q-A pair curation complete")
    logger.info(f"Full training set: {len(train_examples)} examples")
    logger.info(f"Small training set: {len(small_sampled)} examples")
    logger.info(f"Test set: {len(test_examples)} examples")

if __name__ == "__main__":
    main()
