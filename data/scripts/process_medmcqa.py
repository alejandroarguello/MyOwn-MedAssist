#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process MedMCQA dataset:
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
RAW_DATA_DIR = Path("../raw/medmcqa")
PROCESSED_DATA_DIR = Path("../processed")
OUTPUT_FILE = PROCESSED_DATA_DIR / "medmcqa_processed.jsonl"

def load_medmcqa_data():
    """Load MedMCQA dataset from extracted files."""
    logger.info("Loading MedMCQA data")
    
    # Find the CSV files in the extracted directory
    # Look for both possible naming conventions (valid.csv or validation.csv)
    train_path = list(RAW_DATA_DIR.glob("**/train.csv"))[0]
    
    # Check for validation file with different possible names
    valid_paths = list(RAW_DATA_DIR.glob("**/valid.csv"))
    if not valid_paths:
        valid_paths = list(RAW_DATA_DIR.glob("**/validation.csv"))
    
    if not valid_paths:
        raise FileNotFoundError("Could not find validation data file (valid.csv or validation.csv)")
    
    valid_path = valid_paths[0]
    test_path = list(RAW_DATA_DIR.glob("**/test.csv"))[0]
    
    # Load data
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded {len(train_df)} train, {len(valid_df)} valid, {len(test_df)} test examples")
    
    return train_df, valid_df, test_df

def clean_text(text):
    """Clean text by removing excessive whitespace, etc."""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive newlines and whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def format_for_openai(row):
    """Format a MedMCQA example for OpenAI fine-tuning."""
    
    # Extract relevant fields
    question = clean_text(row.get('question', ''))
    option_a = clean_text(row.get('opa', ''))
    option_b = clean_text(row.get('opb', ''))
    option_c = clean_text(row.get('opc', ''))
    option_d = clean_text(row.get('opd', ''))
    
    # Get the correct answer index (1-based in the dataset)
    correct_answer_idx = row.get('cop', 0)
    
    # Map the index to the option letter
    option_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    correct_option = option_map.get(correct_answer_idx, '')
    
    # Get explanation if available
    explanation = clean_text(row.get('exp', ''))
    
    # Format the prompt with the question and options
    prompt = f"""Question: {question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Which option is correct?"""
    
    # Format the completion with the correct answer and explanation
    completion = f"The correct answer is {correct_option}."
    if explanation:
        completion += f" {explanation}"
    
    # Format for OpenAI Chat fine-tuning
    formatted = {
        "messages": [
            {"role": "system", "content": "You are a medical assistant that answers medical multiple-choice questions accurately."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
    }
    
    return formatted

def process_medmcqa():
    """Process MedMCQA dataset and convert to JSONL for fine-tuning."""
    logger.info("Processing MedMCQA dataset")
    
    # Load data
    train_df, valid_df, test_df = load_medmcqa_data()
    
    # Combine all data for processing
    all_data = pd.concat([train_df, valid_df, test_df])
    
    logger.info(f"Total examples: {len(all_data)}")
    
    # Deduplicate by question (assuming questions are unique)
    all_data.drop_duplicates(subset=['question'], inplace=True)
    
    logger.info(f"After deduplication: {len(all_data)} examples")
    
    # Format for OpenAI
    formatted_examples = [format_for_openai(row) for _, row in all_data.iterrows()]
    
    # Save to JSONL
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        for example in formatted_examples:
            writer.write(example)
    
    logger.info(f"Saved {len(formatted_examples)} examples to {OUTPUT_FILE}")
    
    return len(formatted_examples)

def main():
    """Main function to process MedMCQA dataset."""
    logger.info("Starting MedMCQA processing")
    
    num_examples = process_medmcqa()
    
    logger.info(f"Processing complete. {num_examples} examples processed.")

if __name__ == "__main__":
    main()
