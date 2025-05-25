#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified script to analyze and visualize the processed datasets:
1. PubMedQA
2. MedMCQA
3. Synthea EHR

This script generates basic statistics and visualizations without relying on NLTK.
"""

import os
import json
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "data_analysis/visualizations"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    data = []
    if not Path(file_path).exists():
        print(f"Warning: File {file_path} does not exist.")
        return data
    
    try:
        with jsonlines.open(file_path) as reader:
            for item in reader:
                data.append(item)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data

def extract_text_from_messages(data):
    """Extract text from the messages field in the dataset."""
    questions = []
    answers = []
    system_prompts = []
    
    for item in data:
        messages = item.get('messages', [])
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if role == 'user':
                questions.append(content)
            elif role == 'assistant':
                answers.append(content)
            elif role == 'system':
                system_prompts.append(content)
    
    return questions, answers, system_prompts

def simple_word_count(text):
    """Simple word count function that splits on whitespace."""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())

def count_words(text_list):
    """Count words in a list of texts using simple splitting."""
    word_counts = []
    for text in text_list:
        word_counts.append(simple_word_count(text))
    return word_counts if word_counts else [0]  # Return [0] if empty to avoid numpy errors

def extract_medical_entities(text_list):
    """Extract common medical terms from text."""
    medical_terms = []
    # Simple regex pattern for medical terms
    medical_patterns = [
        r'\b(?:disease|syndrome|disorder|infection|cancer|tumor|virus|bacteria|treatment|therapy|medication|drug|diagnosis|symptom|patient|doctor|hospital|surgery|condition|chronic|acute)\b',
        r'\b(?:diabetes|hypertension|asthma|arthritis|depression|anxiety|obesity|pneumonia|influenza|covid)\b'
    ]
    
    for text in text_list:
        if not isinstance(text, str):
            continue
        for pattern in medical_patterns:
            matches = re.findall(pattern, text.lower())
            medical_terms.extend(matches)
    
    return Counter(medical_terms)

def analyze_pubmedqa():
    """Analyze PubMedQA dataset."""
    print("Analyzing PubMedQA dataset...")
    
    # Load data
    file_path = PROCESSED_DIR / "pubmedqa_processed.jsonl"
    data = load_jsonl(file_path)
    
    if not data:
        print("No PubMedQA data found.")
        return None
    
    # Extract text
    questions, answers, system_prompts = extract_text_from_messages(data)
    
    # Basic statistics
    stats = {
        "Total examples": len(data),
        "Unique system prompts": len(set(system_prompts)),
        "Average question length (words)": np.mean(count_words(questions)),
        "Average answer length (words)": np.mean(count_words(answers)),
        "Max question length (words)": max(count_words(questions)),
        "Max answer length (words)": max(count_words(answers)),
    }
    
    # Extract medical entities
    medical_terms = extract_medical_entities(questions + answers)
    
    # Create visualizations
    
    # 1. Question and answer length distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(count_words(questions), kde=True)
    plt.title('PubMedQA: Question Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(count_words(answers), kde=True)
    plt.title('PubMedQA: Answer Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pubmedqa_length_distributions.png")
    plt.close()
    
    # 2. Top medical terms
    plt.figure(figsize=(12, 6))
    top_terms = dict(medical_terms.most_common(15))
    sns.barplot(x=list(top_terms.keys()), y=list(top_terms.values()))
    plt.title('PubMedQA: Top Medical Terms')
    plt.xlabel('Term')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pubmedqa_top_medical_terms.png")
    plt.close()
    
    return stats

def analyze_medmcqa():
    """Analyze MedMCQA dataset."""
    print("Analyzing MedMCQA dataset...")
    
    # Load data
    file_path = PROCESSED_DIR / "medmcqa_processed.jsonl"
    data = load_jsonl(file_path)
    
    if not data:
        print("No MedMCQA data found.")
        return None
    
    # Extract text
    questions, answers, system_prompts = extract_text_from_messages(data)
    
    # Extract option patterns from questions
    option_pattern = re.compile(r'Options:\s*\n([A-D]\..*\n)+', re.MULTILINE)
    options_present = sum(1 for q in questions if isinstance(q, str) and option_pattern.search(q))
    
    # Basic statistics
    stats = {
        "Total examples": len(data),
        "Unique system prompts": len(set(system_prompts)),
        "Questions with options": options_present,
        "Percentage with options": options_present / len(questions) * 100 if questions else 0,
        "Average question length (words)": np.mean(count_words(questions)),
        "Average answer length (words)": np.mean(count_words(answers)),
        "Max question length (words)": max(count_words(questions)),
        "Max answer length (words)": max(count_words(answers)),
    }
    
    # Extract medical entities
    medical_terms = extract_medical_entities(questions + answers)
    
    # Create visualizations
    
    # 1. Question and answer length distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(count_words(questions), kde=True)
    plt.title('MedMCQA: Question Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(count_words(answers), kde=True)
    plt.title('MedMCQA: Answer Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "medmcqa_length_distributions.png")
    plt.close()
    
    # 2. Top medical terms
    plt.figure(figsize=(12, 6))
    top_terms = dict(medical_terms.most_common(15))
    sns.barplot(x=list(top_terms.keys()), y=list(top_terms.values()))
    plt.title('MedMCQA: Top Medical Terms')
    plt.xlabel('Term')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "medmcqa_top_medical_terms.png")
    plt.close()
    
    return stats

def analyze_synthea():
    """Analyze Synthea dataset."""
    print("Analyzing Synthea dataset...")
    
    # Load data
    file_path = PROCESSED_DIR / "synthea_curated.jsonl"
    data = load_jsonl(file_path)
    
    if not data:
        print("No Synthea data found.")
        return None
    
    # Extract text
    questions, answers, system_prompts = extract_text_from_messages(data)
    
    # Extract conditions from questions
    condition_pattern = re.compile(r'What medications might be prescribed for (.*?)\?')
    conditions = []
    for q in questions:
        if not isinstance(q, str):
            continue
        match = condition_pattern.search(q)
        if match:
            conditions.append(match.group(1))
    
    # Extract medications from answers
    medication_pattern = re.compile(r'medications that might be prescribed include: (.*?)\.', re.DOTALL)
    medications_lists = []
    for a in answers:
        if not isinstance(a, str):
            continue
        match = medication_pattern.search(a)
        if match:
            meds = match.group(1).split(',')
            medications_lists.extend([m.strip() for m in meds])
    
    # Basic statistics
    stats = {
        "Total examples": len(data),
        "Unique system prompts": len(set(system_prompts)),
        "Unique conditions": len(set(conditions)),
        "Unique medications": len(set(medications_lists)),
        "Average question length (words)": np.mean(count_words(questions)),
        "Average answer length (words)": np.mean(count_words(answers)),
        "Max question length (words)": max(count_words(questions)),
        "Max answer length (words)": max(count_words(answers)),
    }
    
    # Count condition frequencies
    condition_counts = Counter(conditions)
    
    # Create visualizations
    
    # 1. Question and answer length distributions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(count_words(questions), kde=True)
    plt.title('Synthea: Question Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(count_words(answers), kde=True)
    plt.title('Synthea: Answer Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "synthea_length_distributions.png")
    plt.close()
    
    # 2. Top conditions
    if conditions:
        plt.figure(figsize=(12, 8))
        top_conditions = dict(condition_counts.most_common(15))
        sns.barplot(x=list(top_conditions.values()), y=list(top_conditions.keys()))
        plt.title('Synthea: Top Medical Conditions')
        plt.xlabel('Frequency')
        plt.ylabel('Condition')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "synthea_top_conditions.png")
        plt.close()
    
    return stats

def compare_datasets(pubmedqa_stats, medmcqa_stats, synthea_stats):
    """Compare statistics across datasets."""
    print("Comparing datasets...")
    
    # Prepare comparison data
    available_datasets = []
    total_examples = []
    avg_q_length = []
    avg_a_length = []
    
    # Add data for available datasets
    if pubmedqa_stats is not None:
        available_datasets.append('PubMedQA')
        total_examples.append(pubmedqa_stats["Total examples"])
        avg_q_length.append(pubmedqa_stats["Average question length (words)"])
        avg_a_length.append(pubmedqa_stats["Average answer length (words)"])
    
    if medmcqa_stats is not None:
        available_datasets.append('MedMCQA')
        total_examples.append(medmcqa_stats["Total examples"])
        avg_q_length.append(medmcqa_stats["Average question length (words)"])
        avg_a_length.append(medmcqa_stats["Average answer length (words)"])
    
    if synthea_stats is not None:
        available_datasets.append('Synthea')
        total_examples.append(synthea_stats["Total examples"])
        avg_q_length.append(synthea_stats["Average question length (words)"])
        avg_a_length.append(synthea_stats["Average answer length (words)"])
    
    if len(available_datasets) < 2:
        print("Not enough datasets available for comparison.")
        return
    
    # Create visualizations
    
    # 1. Total examples comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=available_datasets, y=total_examples)
    plt.title('Dataset Size Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Number of Examples')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_size_comparison.png")
    plt.close()
    
    # 2. Average question and answer length comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(available_datasets))
    width = 0.35
    
    plt.bar(x - width/2, avg_q_length, width, label='Avg Question Length')
    plt.bar(x + width/2, avg_a_length, width, label='Avg Answer Length')
    
    plt.title('Average Text Length Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Average Word Count')
    plt.xticks(x, available_datasets)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "text_length_comparison.png")
    plt.close()
    
    # 3. Create a summary table
    summary = pd.DataFrame({
        'Dataset': available_datasets,
        'Total Examples': total_examples,
        'Avg Question Length': [round(val, 2) for val in avg_q_length],
        'Avg Answer Length': [round(val, 2) for val in avg_a_length],
    })
    
    # Save summary to CSV
    summary.to_csv(OUTPUT_DIR / "dataset_comparison_summary.csv", index=False)
    
    # Print summary
    print("\nDataset Comparison Summary:")
    print(summary)

def main():
    """Main function to analyze all datasets."""
    print("Starting dataset analysis...")
    
    # Check if processed directory exists
    if not PROCESSED_DIR.exists():
        print(f"Warning: Processed data directory {PROCESSED_DIR} does not exist.")
        print("Please run the data processing pipeline first to generate the processed datasets.")
        return
    
    # Analyze each dataset
    pubmedqa_file = PROCESSED_DIR / "pubmedqa_processed.jsonl"
    medmcqa_file = PROCESSED_DIR / "medmcqa_processed.jsonl"
    synthea_file = PROCESSED_DIR / "synthea_curated.jsonl"
    
    pubmedqa_stats = None
    medmcqa_stats = None
    synthea_stats = None
    
    if pubmedqa_file.exists():
        print("Analyzing PubMedQA dataset...")
        pubmedqa_stats = analyze_pubmedqa()
    else:
        print("PubMedQA dataset not found. Skipping analysis.")
    
    if medmcqa_file.exists():
        print("Analyzing MedMCQA dataset...")
        medmcqa_stats = analyze_medmcqa()
    else:
        print("MedMCQA dataset not found. Skipping analysis.")
    
    if synthea_file.exists():
        print("Analyzing Synthea dataset...")
        synthea_stats = analyze_synthea()
    else:
        print("Synthea dataset not found. Skipping analysis.")
    
    # Check if we have at least two datasets to compare
    datasets_available = [stats for stats in [pubmedqa_stats, medmcqa_stats, synthea_stats] if stats is not None]
    if len(datasets_available) >= 2:
        print("Comparing available datasets...")
        compare_datasets(pubmedqa_stats, medmcqa_stats, synthea_stats)
    else:
        print("Not enough datasets available for comparison.")
    
    print("\nAnalysis complete. Visualizations saved to:", OUTPUT_DIR)
    
    # Create a summary report
    with open(OUTPUT_DIR / "analysis_summary.txt", "w") as f:
        f.write("# Medical Dataset Analysis Summary\n\n")
        
        if pubmedqa_stats:
            f.write("## PubMedQA Dataset\n")
            for key, value in pubmedqa_stats.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
        else:
            f.write("## PubMedQA Dataset\n")
            f.write("- Dataset not available for analysis\n\n")
        
        if medmcqa_stats:
            f.write("## MedMCQA Dataset\n")
            for key, value in medmcqa_stats.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
        else:
            f.write("## MedMCQA Dataset\n")
            f.write("- Dataset not available for analysis\n\n")
        
        if synthea_stats:
            f.write("## Synthea Dataset\n")
            for key, value in synthea_stats.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")
        else:
            f.write("## Synthea Dataset\n")
            f.write("- Dataset not available for analysis\n\n")
        
        f.write("Visualizations are available in the 'visualizations' directory.\n")

if __name__ == "__main__":
    main()
