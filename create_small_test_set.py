#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a smaller test set for evaluation.
This script randomly selects 150 samples from the original test set.
"""

import json
import random
from pathlib import Path

# Input and output paths
INPUT_FILE = Path("fine_tuning/data/test.jsonl")
OUTPUT_FILE = Path("fine_tuning/data/test_small.jsonl")

# Number of samples to select
NUM_SAMPLES = 150

def main():
    """Create a smaller test set by randomly sampling the original test set."""
    print(f"Reading data from {INPUT_FILE}")
    
    # Read all samples from the original test set
    samples = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Read {len(samples)} samples from original test set")
    
    # Randomly select samples
    if len(samples) <= NUM_SAMPLES:
        selected_samples = samples
        print("Original test set is smaller than or equal to requested size, using all samples")
    else:
        selected_samples = random.sample(samples, NUM_SAMPLES)
        print(f"Randomly selected {NUM_SAMPLES} samples")
    
    # Write selected samples to the output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Wrote {len(selected_samples)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
