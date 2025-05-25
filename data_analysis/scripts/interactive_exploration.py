#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interactive exploration script for medical datasets.
This script provides functions to interactively explore the datasets
and can be used in a Python interactive session or Jupyter notebook.
"""

import json
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR = PROJECT_ROOT / "data_analysis/visualizations"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

class DatasetExplorer:
    """Class for exploring medical datasets."""
    
    def __init__(self):
        self.datasets = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_dataset(self, name, file_path):
        """Load a dataset from a JSONL file."""
        print(f"Loading {name} dataset...")
        data = []
        with jsonlines.open(file_path) as reader:
            for item in reader:
                data.append(item)
        
        self.datasets[name] = data
        print(f"Loaded {len(data)} examples from {name}")
        return data
    
    def load_all_datasets(self):
        """Load all three datasets."""
        self.load_dataset("PubMedQA", PROCESSED_DIR / "pubmedqa_processed.jsonl")
        self.load_dataset("MedMCQA", PROCESSED_DIR / "medmcqa_processed.jsonl")
        self.load_dataset("Synthea", PROCESSED_DIR / "synthea_curated.jsonl")
    
    def extract_messages(self, dataset_name):
        """Extract messages from a dataset."""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not loaded. Please load it first.")
            return None, None, None
        
        data = self.datasets[dataset_name]
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
    
    def get_basic_stats(self, dataset_name):
        """Get basic statistics for a dataset."""
        questions, answers, system_prompts = self.extract_messages(dataset_name)
        if questions is None:
            return None
        
        # Count words
        q_word_counts = [len(word_tokenize(q)) for q in questions]
        a_word_counts = [len(word_tokenize(a)) for a in answers]
        
        stats = {
            "Total examples": len(self.datasets[dataset_name]),
            "Unique system prompts": len(set(system_prompts)),
            "Average question length (words)": np.mean(q_word_counts),
            "Average answer length (words)": np.mean(a_word_counts),
            "Median question length (words)": np.median(q_word_counts),
            "Median answer length (words)": np.median(a_word_counts),
            "Max question length (words)": max(q_word_counts),
            "Max answer length (words)": max(a_word_counts),
            "Min question length (words)": min(q_word_counts),
            "Min answer length (words)": min(a_word_counts),
        }
        
        return stats
    
    def plot_length_distributions(self, dataset_name):
        """Plot length distributions for a dataset."""
        questions, answers, _ = self.extract_messages(dataset_name)
        if questions is None:
            return
        
        # Count words
        q_word_counts = [len(word_tokenize(q)) for q in questions]
        a_word_counts = [len(word_tokenize(a)) for a in answers]
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(q_word_counts, kde=True, bins=30)
        plt.title(f'{dataset_name}: Question Length Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(q_word_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(q_word_counts):.1f}')
        plt.axvline(np.median(q_word_counts), color='green', linestyle='--', 
                   label=f'Median: {np.median(q_word_counts):.1f}')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.histplot(a_word_counts, kde=True, bins=30)
        plt.title(f'{dataset_name}: Answer Length Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(a_word_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(a_word_counts):.1f}')
        plt.axvline(np.median(a_word_counts), color='green', linestyle='--', 
                   label=f'Median: {np.median(a_word_counts):.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def extract_medical_terms(self, text_list, n_terms=20):
        """Extract and count medical terms from text."""
        # Simple regex pattern for medical terms
        medical_patterns = [
            r'\b(?:disease|syndrome|disorder|infection|cancer|tumor|virus|bacteria|treatment|therapy|medication|drug|diagnosis|symptom|patient|doctor|hospital|surgery|condition|chronic|acute)\b',
            r'\b(?:diabetes|hypertension|asthma|arthritis|depression|anxiety|obesity|pneumonia|influenza|covid)\b'
        ]
        
        medical_terms = []
        for text in text_list:
            for pattern in medical_patterns:
                matches = re.findall(pattern, text.lower())
                medical_terms.extend(matches)
        
        return Counter(medical_terms).most_common(n_terms)
    
    def plot_top_terms(self, dataset_name, n_terms=15):
        """Plot top medical terms for a dataset."""
        questions, answers, _ = self.extract_messages(dataset_name)
        if questions is None:
            return
        
        # Extract medical terms
        q_terms = self.extract_medical_terms(questions, n_terms)
        a_terms = self.extract_medical_terms(answers, n_terms)
        
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 1, 1)
        sns.barplot(x=[term[0] for term in q_terms], y=[term[1] for term in q_terms])
        plt.title(f'{dataset_name}: Top Medical Terms in Questions')
        plt.xlabel('Term')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(2, 1, 2)
        sns.barplot(x=[term[0] for term in a_terms], y=[term[1] for term in a_terms])
        plt.title(f'{dataset_name}: Top Medical Terms in Answers')
        plt.xlabel('Term')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def create_wordcloud(self, text_list, title):
        """Create and display a word cloud."""
        # Combine all texts
        all_text = ' '.join(text_list)
        
        # Tokenize and filter stop words
        words = word_tokenize(all_text.lower())
        filtered_words = [self.lemmatizer.lemmatize(word) for word in words 
                         if word.isalpha() and word not in self.stop_words]
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate(' '.join(filtered_words))
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_wordclouds(self, dataset_name):
        """Plot word clouds for a dataset."""
        questions, answers, _ = self.extract_messages(dataset_name)
        if questions is None:
            return
        
        self.create_wordcloud(questions, f'{dataset_name}: Question Word Cloud')
        self.create_wordcloud(answers, f'{dataset_name}: Answer Word Cloud')
    
    def compare_datasets(self):
        """Compare statistics across datasets."""
        if len(self.datasets) < 2:
            print("Please load at least two datasets to compare.")
            return
        
        # Prepare comparison data
        dataset_names = list(self.datasets.keys())
        stats = {}
        
        for name in dataset_names:
            stats[name] = self.get_basic_stats(name)
        
        # Create comparison dataframe
        comparison = pd.DataFrame({
            'Dataset': dataset_names,
            'Total Examples': [stats[name]['Total examples'] for name in dataset_names],
            'Avg Question Length': [round(stats[name]['Average question length (words)'], 2) 
                                  for name in dataset_names],
            'Avg Answer Length': [round(stats[name]['Average answer length (words)'], 2) 
                                for name in dataset_names],
            'Median Question Length': [round(stats[name]['Median question length (words)'], 2) 
                                     for name in dataset_names],
            'Median Answer Length': [round(stats[name]['Median answer length (words)'], 2) 
                                   for name in dataset_names],
        })
        
        # Plot comparisons
        plt.figure(figsize=(12, 6))
        
        # Dataset size comparison
        plt.subplot(1, 2, 1)
        sns.barplot(x='Dataset', y='Total Examples', data=comparison)
        plt.title('Dataset Size Comparison')
        plt.xticks(rotation=45, ha='right')
        
        # Text length comparison
        plt.subplot(1, 2, 2)
        x = np.arange(len(dataset_names))
        width = 0.35
        
        plt.bar(x - width/2, comparison['Avg Question Length'], width, label='Avg Question Length')
        plt.bar(x + width/2, comparison['Avg Answer Length'], width, label='Avg Answer Length')
        
        plt.title('Average Text Length Comparison')
        plt.xlabel('Dataset')
        plt.ylabel('Average Word Count')
        plt.xticks(x, dataset_names, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return comparison
    
    def create_vectorized_representation(self, dataset_name, text_type='questions', method='tfidf'):
        """Create a vectorized representation of text for visualization."""
        questions, answers, _ = self.extract_messages(dataset_name)
        if questions is None:
            return None
        
        # Choose text to vectorize
        if text_type == 'questions':
            texts = questions
        elif text_type == 'answers':
            texts = answers
        else:
            print("Invalid text_type. Choose 'questions' or 'answers'.")
            return None
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            # Tokenize
            tokens = word_tokenize(text.lower())
            # Remove stop words and non-alphabetic tokens
            filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                              if token.isalpha() and token not in self.stop_words]
            processed_texts.append(' '.join(filtered_tokens))
        
        # Vectorize
        if method == 'count':
            vectorizer = CountVectorizer(max_features=1000)
        elif method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=1000)
        else:
            print("Invalid method. Choose 'count' or 'tfidf'.")
            return None
        
        X = vectorizer.fit_transform(processed_texts)
        
        return X, vectorizer.get_feature_names_out()
    
    def visualize_embeddings(self, dataset_name, text_type='questions', method='tfidf', 
                            dim_reduction='tsne', n_components=2):
        """Visualize text embeddings using dimensionality reduction."""
        X, feature_names = self.create_vectorized_representation(dataset_name, text_type, method)
        if X is None:
            return
        
        # Apply dimensionality reduction
        if dim_reduction == 'pca':
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X.toarray())
        elif dim_reduction == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X.toarray())
        elif dim_reduction == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X)
        else:
            print("Invalid dim_reduction. Choose 'pca', 'tsne', or 'svd'.")
            return
        
        # Create a dataframe for plotting
        df = pd.DataFrame(X_reduced, columns=[f'Component {i+1}' for i in range(n_components)])
        
        # Add text length as a feature
        questions, answers, _ = self.extract_messages(dataset_name)
        if text_type == 'questions':
            df['Text Length'] = [len(word_tokenize(q)) for q in questions]
        else:
            df['Text Length'] = [len(word_tokenize(a)) for a in answers]
        
        # Create interactive plot with Plotly
        if n_components == 2:
            fig = px.scatter(df, x='Component 1', y='Component 2', 
                           color='Text Length', title=f'{dataset_name} {text_type.capitalize()} Embeddings')
        elif n_components == 3:
            fig = px.scatter_3d(df, x='Component 1', y='Component 2', z='Component 3',
                              color='Text Length', title=f'{dataset_name} {text_type.capitalize()} Embeddings')
        
        fig.show()
        
        return df, X_reduced
    
    def analyze_synthea_specific(self):
        """Perform Synthea-specific analysis."""
        if "Synthea" not in self.datasets:
            print("Synthea dataset not loaded. Please load it first.")
            return
        
        questions, answers, _ = self.extract_messages("Synthea")
        
        # Extract conditions from questions
        condition_pattern = re.compile(r'What medications might be prescribed for (.*?)\?')
        conditions = []
        for q in questions:
            match = condition_pattern.search(q)
            if match:
                conditions.append(match.group(1))
        
        # Extract medications from answers
        medication_pattern = re.compile(r'medications that might be prescribed include: (.*?)\.', re.DOTALL)
        medications_by_condition = defaultdict(list)
        
        for i, a in enumerate(answers):
            if i < len(conditions):  # Ensure we have a condition for this answer
                match = medication_pattern.search(a)
                if match:
                    meds = [m.strip() for m in match.group(1).split(',')]
                    medications_by_condition[conditions[i]].extend(meds)
        
        # Count condition frequencies
        condition_counts = Counter(conditions)
        
        # Plot top conditions
        plt.figure(figsize=(12, 8))
        top_conditions = dict(condition_counts.most_common(15))
        sns.barplot(x=list(top_conditions.values()), y=list(top_conditions.keys()))
        plt.title('Synthea: Top Medical Conditions')
        plt.xlabel('Frequency')
        plt.ylabel('Condition')
        plt.tight_layout()
        plt.show()
        
        # Create a network visualization of conditions and medications
        # (This would be better with an interactive tool like NetworkX + Plotly)
        
        return condition_counts, medications_by_condition
    
    def analyze_medmcqa_specific(self):
        """Perform MedMCQA-specific analysis."""
        if "MedMCQA" not in self.datasets:
            print("MedMCQA dataset not loaded. Please load it first.")
            return
        
        questions, answers, _ = self.extract_messages("MedMCQA")
        
        # Extract options from questions
        option_pattern = re.compile(r'Options:\s*\n([A-D]\..*\n)+', re.MULTILINE)
        options_present = sum(1 for q in questions if option_pattern.search(q))
        
        # Extract answer choice patterns
        answer_pattern = re.compile(r'The correct answer is ([A-D])', re.IGNORECASE)
        answer_choices = []
        
        for a in answers:
            match = answer_pattern.search(a)
            if match:
                answer_choices.append(match.group(1).upper())
        
        # Count answer choices
        choice_counts = Counter(answer_choices)
        
        # Plot answer choice distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(choice_counts.keys()), y=list(choice_counts.values()))
        plt.title('MedMCQA: Answer Choice Distribution')
        plt.xlabel('Answer Choice')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        return choice_counts
    
    def analyze_pubmedqa_specific(self):
        """Perform PubMedQA-specific analysis."""
        if "PubMedQA" not in self.datasets:
            print("PubMedQA dataset not loaded. Please load it first.")
            return
        
        questions, answers, _ = self.extract_messages("PubMedQA")
        
        # Extract yes/no/maybe answers
        yes_count = sum(1 for a in answers if a.lower().strip() == 'yes')
        no_count = sum(1 for a in answers if a.lower().strip() == 'no')
        maybe_count = sum(1 for a in answers if a.lower().strip() == 'maybe')
        other_count = len(answers) - yes_count - no_count - maybe_count
        
        # Plot answer type distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=['Yes', 'No', 'Maybe', 'Other'], 
                   y=[yes_count, no_count, maybe_count, other_count])
        plt.title('PubMedQA: Answer Type Distribution')
        plt.xlabel('Answer Type')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        # Extract context patterns
        context_pattern = re.compile(r'Context: (.*?)\n\nQuestion:', re.DOTALL)
        context_lengths = []
        
        for q in questions:
            match = context_pattern.search(q)
            if match:
                context = match.group(1)
                context_lengths.append(len(word_tokenize(context)))
        
        # Plot context length distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(context_lengths, kde=True, bins=30)
        plt.title('PubMedQA: Context Length Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(context_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(context_lengths):.1f}')
        plt.axvline(np.median(context_lengths), color='green', linestyle='--', 
                   label=f'Median: {np.median(context_lengths):.1f}')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return {'yes': yes_count, 'no': no_count, 'maybe': maybe_count, 'other': other_count}
    
    def sample_examples(self, dataset_name, n=5):
        """Sample and display examples from a dataset."""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not loaded. Please load it first.")
            return
        
        data = self.datasets[dataset_name]
        samples = np.random.choice(data, min(n, len(data)), replace=False)
        
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1} from {dataset_name} ---")
            for msg in sample.get('messages', []):
                role = msg.get('role', '')
                content = msg.get('content', '')
                print(f"\n{role.upper()}:\n{content}")
            print("\n" + "-"*50)
        
        return samples

# Example usage
if __name__ == "__main__":
    print("Interactive Medical Dataset Explorer")
    print("This script provides functions to explore the datasets.")
    print("Import this module in your Python session or notebook to use it.")
    print("\nExample usage:")
    print("explorer = DatasetExplorer()")
    print("explorer.load_all_datasets()")
    print("explorer.get_basic_stats('PubMedQA')")
    print("explorer.plot_length_distributions('MedMCQA')")
    print("explorer.compare_datasets()")
