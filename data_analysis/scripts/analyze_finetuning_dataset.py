#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze the fine-tuning dataset composition and distribution.
This helps understand how the ~2000 Q-A pairs are distributed across
the different source datasets (PubMedQA, MedMCQA, Synthea).
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
FINETUNING_DIR = PROJECT_ROOT / "data/finetuning"
OUTPUT_DIR = PROJECT_ROOT / "data_analysis/visualizations"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with jsonlines.open(file_path) as reader:
        for item in reader:
            data.append(item)
    return data

def detect_dataset_source(example):
    """
    Detect the source dataset of an example based on its content patterns.
    Returns: 'pubmedqa', 'medmcqa', 'synthea', or 'unknown'
    """
    # Get the user message (question)
    user_message = ""
    for msg in example.get('messages', []):
        if msg.get('role') == 'user':
            user_message = msg.get('content', '')
            break
    
    # Check for dataset-specific patterns
    if "Context:" in user_message and "Question:" in user_message:
        return "pubmedqa"
    elif "Options:" in user_message and any(f"{opt}." in user_message for opt in "ABCD"):
        return "medmcqa"
    elif "What medications might be prescribed for" in user_message:
        return "synthea"
    else:
        return "unknown"

def analyze_finetuning_dataset():
    """Analyze the fine-tuning dataset composition."""
    # Check if fine-tuning dataset exists
    finetuning_file = FINETUNING_DIR / "finetuning_data.jsonl"
    if not finetuning_file.exists():
        # If not, check processed files to simulate what would be in the fine-tuning dataset
        pubmedqa_file = PROCESSED_DIR / "pubmedqa_processed.jsonl"
        medmcqa_file = PROCESSED_DIR / "medmcqa_processed.jsonl"
        synthea_file = PROCESSED_DIR / "synthea_curated.jsonl"
        
        all_data = []
        dataset_sources = []
        
        if pubmedqa_file.exists():
            pubmedqa_data = load_jsonl(pubmedqa_file)
            all_data.extend(pubmedqa_data[:700])  # Assuming ~700 examples from PubMedQA
            dataset_sources.extend(['pubmedqa'] * len(pubmedqa_data[:700]))
        
        if medmcqa_file.exists():
            medmcqa_data = load_jsonl(medmcqa_file)
            all_data.extend(medmcqa_data[:700])  # Assuming ~700 examples from MedMCQA
            dataset_sources.extend(['medmcqa'] * len(medmcqa_data[:700]))
        
        if synthea_file.exists():
            synthea_data = load_jsonl(synthea_file)
            all_data.extend(synthea_data[:600])  # Assuming ~600 examples from Synthea
            dataset_sources.extend(['synthea'] * len(synthea_data[:600]))
    else:
        # Load the actual fine-tuning dataset
        all_data = load_jsonl(finetuning_file)
        # Detect source dataset for each example
        dataset_sources = [detect_dataset_source(example) for example in all_data]
    
    # Count dataset sources
    source_counts = Counter(dataset_sources)
    
    # Calculate percentages
    total = len(all_data)
    percentages = {source: (count / total) * 100 for source, count in source_counts.items()}
    
    # Create visualizations
    
    # 1. Dataset composition pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(
        source_counts.values(), 
        labels=[f"{source.capitalize()}: {count} ({percentages[source]:.1f}%)" 
                for source, count in source_counts.items()],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(source_counts),
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    plt.title('Fine-tuning Dataset Composition', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "finetuning_dataset_composition.png")
    plt.close()
    
    # 2. Dataset composition bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=list(source_counts.keys()),
        y=list(source_counts.values()),
        palette='viridis'
    )
    plt.title('Fine-tuning Dataset Composition', fontsize=16)
    plt.xlabel('Source Dataset', fontsize=14)
    plt.ylabel('Number of Examples', fontsize=14)
    
    # Add count and percentage labels on top of bars
    for i, (source, count) in enumerate(source_counts.items()):
        plt.text(
            i, 
            count + 10, 
            f"{count} ({percentages[source]:.1f}%)",
            ha='center',
            fontsize=12
        )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "finetuning_dataset_composition_bar.png")
    plt.close()
    
    # Analyze text lengths by dataset source
    question_lengths = []
    answer_lengths = []
    sources = []
    
    for example, source in zip(all_data, dataset_sources):
        q_length = 0
        a_length = 0
        
        for msg in example.get('messages', []):
            if msg.get('role') == 'user':
                q_length = len(msg.get('content', '').split())
            elif msg.get('role') == 'assistant':
                a_length = len(msg.get('content', '').split())
        
        question_lengths.append(q_length)
        answer_lengths.append(a_length)
        sources.append(source)
    
    # Create dataframe for analysis
    df = pd.DataFrame({
        'Source': sources,
        'QuestionLength': question_lengths,
        'AnswerLength': answer_lengths
    })
    
    # 3. Question length by dataset source
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Source', y='QuestionLength', data=df, palette='viridis')
    plt.title('Question Length by Dataset Source', fontsize=16)
    plt.xlabel('Source Dataset', fontsize=14)
    plt.ylabel('Question Length (words)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "finetuning_question_length_by_source.png")
    plt.close()
    
    # 4. Answer length by dataset source
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Source', y='AnswerLength', data=df, palette='viridis')
    plt.title('Answer Length by Dataset Source', fontsize=16)
    plt.xlabel('Source Dataset', fontsize=14)
    plt.ylabel('Answer Length (words)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "finetuning_answer_length_by_source.png")
    plt.close()
    
    # 5. Combined length distributions
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(df['QuestionLength'], kde=True, bins=30)
    plt.title('Question Length Distribution (All Sources)', fontsize=14)
    plt.xlabel('Question Length (words)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.subplot(2, 2, 2)
    sns.histplot(df['AnswerLength'], kde=True, bins=30)
    plt.title('Answer Length Distribution (All Sources)', fontsize=14)
    plt.xlabel('Answer Length (words)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.subplot(2, 2, 3)
    for source in df['Source'].unique():
        sns.kdeplot(df[df['Source'] == source]['QuestionLength'], label=source)
    plt.title('Question Length by Source', fontsize=14)
    plt.xlabel('Question Length (words)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for source in df['Source'].unique():
        sns.kdeplot(df[df['Source'] == source]['AnswerLength'], label=source)
    plt.title('Answer Length by Source', fontsize=14)
    plt.xlabel('Answer Length (words)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "finetuning_length_distributions.png")
    plt.close()
    
    # Calculate statistics
    stats = {
        'Total examples': total,
        'Dataset composition': dict(source_counts),
        'Dataset percentages': {k: f"{v:.1f}%" for k, v in percentages.items()},
        'Average question length': df['QuestionLength'].mean(),
        'Average answer length': df['AnswerLength'].mean(),
        'Max question length': df['QuestionLength'].max(),
        'Max answer length': df['AnswerLength'].max(),
        'Average question length by source': df.groupby('Source')['QuestionLength'].mean().to_dict(),
        'Average answer length by source': df.groupby('Source')['AnswerLength'].mean().to_dict(),
    }
    
    # Save statistics to file
    with open(OUTPUT_DIR / "finetuning_dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Create a summary report
    with open(OUTPUT_DIR / "finetuning_analysis_summary.txt", "w") as f:
        f.write("# Fine-tuning Dataset Analysis Summary\n\n")
        
        f.write("## Dataset Composition\n")
        for source, count in source_counts.items():
            f.write(f"- {source.capitalize()}: {count} examples ({percentages[source]:.1f}%)\n")
        f.write(f"- Total: {total} examples\n\n")
        
        f.write("## Text Length Statistics\n")
        f.write(f"- Average question length: {df['QuestionLength'].mean():.1f} words\n")
        f.write(f"- Average answer length: {df['AnswerLength'].mean():.1f} words\n")
        f.write(f"- Maximum question length: {df['QuestionLength'].max()} words\n")
        f.write(f"- Maximum answer length: {df['AnswerLength'].max()} words\n\n")
        
        f.write("## Average Lengths by Source\n")
        for source in df['Source'].unique():
            avg_q = df[df['Source'] == source]['QuestionLength'].mean()
            avg_a = df[df['Source'] == source]['AnswerLength'].mean()
            f.write(f"- {source.capitalize()}:\n")
            f.write(f"  - Questions: {avg_q:.1f} words\n")
            f.write(f"  - Answers: {avg_a:.1f} words\n")
        
        f.write("\nVisualizations are available in the 'visualizations' directory.\n")
    
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")
    return stats

if __name__ == "__main__":
    analyze_finetuning_dataset()
