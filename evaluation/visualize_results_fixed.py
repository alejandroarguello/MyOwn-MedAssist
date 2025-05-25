#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization dashboard for medical QA evaluation results.
Creates interactive plots and tables to analyze model performance.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def load_results(results_path):
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def create_summary_dataframe(results):
    """Create a summary dataframe from results."""
    summary = results['summary']
    
    # Create a list of records
    records = []
    for model, metrics in summary.items():
        record = {
            'Model': model,
            'Factuality': metrics['factuality_mean'],
            'Factuality Std': metrics['factuality_std'],
            'Relevance': metrics['relevance_mean'],
            'Relevance Std': metrics['relevance_std'],
            'Average': metrics['average_mean'],
            'Average Std': metrics['average_std']
        }
        records.append(record)
    
    # Create dataframe
    df = pd.DataFrame(records)
    
    # Rename models for better display
    model_names = {
        'baseline': 'Baseline',
        'baseline_rag': 'Baseline + RAG',
        'fine_tuned': 'Fine-tuned',
        'fine_tuned_rag': 'Fine-tuned + RAG'
    }
    
    df['Model'] = df['Model'].map(model_names)
    
    return df

def create_detailed_dataframes(results):
    """Create detailed dataframes for each metric."""
    # Check if detailed key exists
    if 'detailed' not in results:
        print("Warning: No detailed data found in results file")
        # Create empty dataframes
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    detailed = results['detailed']
    print(f"Found detailed data for {list(detailed.keys())} models")
    
    # Create dataframes for each metric with equal length arrays
    factuality_data = {}
    relevance_data = {}
    
    # Check what's in the detailed data structure
    for model, model_data in detailed.items():
        print(f"Model {model} has keys: {list(model_data.keys())}")
        if 'factuality' in model_data:
            print(f"  Factuality data length: {len(model_data['factuality'])}")
        if 'relevance' in model_data:
            print(f"  Relevance data length: {len(model_data['relevance'])}")
    
    # Get the correct model names
    model_mapping = {
        'baseline': 'Baseline',
        'baseline_rag': 'Baseline + RAG',
        'fine_tuned': 'Fine-tuned',
        'fine_tuned_rag': 'Fine-tuned + RAG'
    }
    
    # Extract factuality and relevance data for each model
    for model, display_name in model_mapping.items():
        if model in detailed:
            # Extract factuality data
            if 'factuality' in detailed[model] and isinstance(detailed[model]['factuality'], list) and len(detailed[model]['factuality']) > 0:
                factuality_data[display_name] = detailed[model]['factuality']
                print(f"Extracted {len(factuality_data[display_name])} factuality scores for {display_name}")
            
            # Extract relevance data
            if 'relevance' in detailed[model] and isinstance(detailed[model]['relevance'], list) and len(detailed[model]['relevance']) > 0:
                relevance_data[display_name] = detailed[model]['relevance']
                print(f"Extracted {len(relevance_data[display_name])} relevance scores for {display_name}")
    
    # Create dataframes (may have different number of rows for each model, which is OK)
    factuality_df = pd.DataFrame(factuality_data)
    relevance_df = pd.DataFrame(relevance_data)
    
    # Calculate average for each example
    average_data = {}
    for display_name in factuality_data.keys():
        if display_name in relevance_data:
            # Need to handle potentially different lengths
            min_len = min(len(factuality_data[display_name]), len(relevance_data[display_name]))
            f_scores = factuality_data[display_name][:min_len]
            r_scores = relevance_data[display_name][:min_len]
            
            # Calculate average
            average = [(f + r) / 2 for f, r in zip(f_scores, r_scores)]
            average_data[display_name] = average
            print(f"Calculated {len(average)} average scores for {display_name}")
    
    average_df = pd.DataFrame(average_data)
    
    # Print dataframe shapes
    print(f"Factuality dataframe shape: {factuality_df.shape}")
    print(f"Relevance dataframe shape: {relevance_df.shape}")
    print(f"Average dataframe shape: {average_df.shape}")
    
    return factuality_df, relevance_df, average_df

def plot_summary_bar_chart(df, output_dir):
    """Create a bar chart of summary results."""
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart
    x = np.arange(len(df['Model']))
    width = 0.25
    
    plt.bar(x - width, df['Factuality'], width, label='Factuality', color='skyblue')
    plt.bar(x, df['Relevance'], width, label='Relevance', color='lightgreen')
    plt.bar(x + width, df['Average'], width, label='Average', color='salmon')
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, df['Model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_dir / 'summary_bar_chart.png')
    plt.close()

def plot_detailed_boxplots(factuality_df, relevance_df, average_df, output_dir):
    """Create boxplots for detailed results."""
    # Filter out empty columns
    factuality_cols = [col for col in factuality_df.columns if factuality_df[col].sum() > 0]
    relevance_cols = [col for col in relevance_df.columns if relevance_df[col].sum() > 0]
    average_cols = [col for col in average_df.columns if average_df[col].sum() > 0]
    
    # Factuality boxplot
    plt.figure(figsize=(12, 8))
    if factuality_cols:
        # Use only non-empty columns
        factuality_df[factuality_cols].boxplot()
        plt.title('Factuality Scores Distribution')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)  # Set y-axis limits to 0-1 range
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No factuality data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title('Factuality Scores Distribution (No Data)')
    plt.tight_layout()
    plt.savefig(output_dir / 'factuality_boxplot.png')
    plt.close()
    
    # Relevance boxplot
    plt.figure(figsize=(12, 8))
    if relevance_cols:
        # Use only non-empty columns
        relevance_df[relevance_cols].boxplot()
        plt.title('Relevance Scores Distribution')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)  # Set y-axis limits to 0-1 range
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No relevance data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title('Relevance Scores Distribution (No Data)')
    plt.tight_layout()
    plt.savefig(output_dir / 'relevance_boxplot.png')
    plt.close()
    
    # Average boxplot
    plt.figure(figsize=(12, 8))
    if average_cols:
        # Use only non-empty columns
        average_df[average_cols].boxplot()
        plt.title('Average Scores Distribution')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)  # Set y-axis limits to 0-1 range
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No average score data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title('Average Scores Distribution (No Data)')
    plt.tight_layout()
    plt.savefig(output_dir / 'average_boxplot.png')
    plt.close()

def plot_score_distributions(factuality_df, relevance_df, average_df, output_dir):
    """Create histograms for score distributions."""
    # Filter out empty columns
    factuality_cols = [col for col in factuality_df.columns if factuality_df[col].sum() > 0]
    relevance_cols = [col for col in relevance_df.columns if relevance_df[col].sum() > 0]
    average_cols = [col for col in average_df.columns if average_df[col].sum() > 0]
    
    # Factuality histograms
    plt.figure(figsize=(12, 8))
    if factuality_cols:
        for column in factuality_cols:
            # Only plot non-zero data
            data = factuality_df[column].dropna()
            if len(data) > 0 and data.max() > 0:
                sns.histplot(data, kde=True, label=column, alpha=0.6)
        plt.xlim(0, 1.1)  # Set x-axis limits to 0-1 range
        plt.title('Factuality Score Distributions')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No factuality data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title('Factuality Score Distributions (No Data)')
    plt.tight_layout()
    plt.savefig(output_dir / 'factuality_distribution.png')
    plt.close()
    
    # Relevance histograms
    plt.figure(figsize=(12, 8))
    if relevance_cols:
        for column in relevance_cols:
            # Only plot non-zero data
            data = relevance_df[column].dropna()
            if len(data) > 0 and data.max() > 0:
                sns.histplot(data, kde=True, label=column, alpha=0.6)
        plt.xlim(0, 1.1)  # Set x-axis limits to 0-1 range
        plt.title('Relevance Score Distributions')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No relevance data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title('Relevance Score Distributions (No Data)')
    plt.tight_layout()
    plt.savefig(output_dir / 'relevance_distribution.png')
    plt.close()
    
    # Average histograms
    plt.figure(figsize=(12, 8))
    if average_cols:
        for column in average_cols:
            # Only plot non-zero data
            data = average_df[column].dropna()
            if len(data) > 0 and data.max() > 0:
                sns.histplot(data, kde=True, label=column, alpha=0.6)
        plt.xlim(0, 1.1)  # Set x-axis limits to 0-1 range
        plt.title('Average Score Distributions')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No average score data available', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
        plt.title('Average Score Distributions (No Data)')
    plt.tight_layout()
    plt.savefig(output_dir / 'average_distribution.png')
    plt.close()

def create_html_report(summary_df, output_dir, image_paths):
    """Create an HTML report with all visualizations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical QA Evaluation Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #2980b9;
                margin-top: 30px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .highlight {
                color: #e74c3c;
                font-weight: bold;
            }
            .visualization {
                margin: 30px 0;
                text-align: center;
            }
            .visualization img {
                max-width: 100%;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .caption {
                margin-top: 10px;
                font-style: italic;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <h1>Medical QA Evaluation Report</h1>
        
        <h2>Summary Results</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Factuality</th>
                <th>Factuality Std</th>
                <th>Relevance</th>
                <th>Relevance Std</th>
                <th>Average</th>
                <th>Average Std</th>
            </tr>
    """
    
    # Add rows for each model
    for _, row in summary_df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['Model']}</td>
                <td>{row['Factuality']:.4f}</td>
                <td>{row['Factuality Std']:.4f}</td>
                <td>{row['Relevance']:.4f}</td>
                <td>{row['Relevance Std']:.4f}</td>
                <td>{row['Average']:.4f}</td>
                <td>{row['Average Std']:.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
    """
    
    # Add visualizations
    for title, path in image_paths.items():
        html_content += f"""
        <div class="visualization">
            <img src="{path.name}" alt="{title}">
            <div class="caption">{title}</div>
        </div>
        """
    
    # Add insights and conclusions
    best_model = summary_df.loc[summary_df['Average'].idxmax()]['Model']
    best_factuality = summary_df.loc[summary_df['Factuality'].idxmax()]['Model']
    best_relevance = summary_df.loc[summary_df['Relevance'].idxmax()]['Model']
    
    html_content += f"""
        <h2>Key Insights</h2>
        <ul>
            <li>The model with the best overall performance is <span class="highlight">{best_model}</span>.</li>
            <li>The model with the highest factuality score is <span class="highlight">{best_factuality}</span>.</li>
            <li>The model with the highest relevance score is <span class="highlight">{best_relevance}</span>.</li>
        </ul>
        
        <h2>Conclusion</h2>
        <p>
            Based on the evaluation results, we can draw the following conclusions:
        </p>
        <ul>
            <li>The baseline model achieves a factuality score of {summary_df[summary_df['Model'] == 'Baseline']['Factuality'].values[0]:.4f} and a relevance score of {summary_df[summary_df['Model'] == 'Baseline']['Relevance'].values[0]:.4f}.</li>
            <li>The fine-tuned model achieves a factuality score of {summary_df[summary_df['Model'] == 'Fine-tuned']['Factuality'].values[0]:.4f} and a relevance score of {summary_df[summary_df['Model'] == 'Fine-tuned']['Relevance'].values[0]:.4f}.</li>
            <li>The difference in performance between the models suggests that {best_model} is the most suitable for medical QA tasks.</li>
        </ul>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_dir / 'evaluation_report.html', 'w') as f:
        f.write(html_content)
    
    return output_dir / 'evaluation_report.html'

def save_detailed_data(factuality_df, relevance_df, average_df, output_dir):
    """Save detailed dataframes to CSV files."""
    # Only save if dataframes are not empty
    if not factuality_df.empty:
        factuality_df.to_csv(output_dir / 'factuality_detailed.csv', index=False)
        print(f"Saved factuality data with shape {factuality_df.shape} to CSV")
    else:
        print("No factuality data to save")
        # Create a minimal CSV with headers
        with open(output_dir / 'factuality_detailed.csv', 'w') as f:
            f.write("Baseline,Baseline + RAG,Fine-tuned,Fine-tuned + RAG\n")
    
    if not relevance_df.empty:
        relevance_df.to_csv(output_dir / 'relevance_detailed.csv', index=False)
        print(f"Saved relevance data with shape {relevance_df.shape} to CSV")
    else:
        print("No relevance data to save")
        # Create a minimal CSV with headers
        with open(output_dir / 'relevance_detailed.csv', 'w') as f:
            f.write("Baseline,Baseline + RAG,Fine-tuned,Fine-tuned + RAG\n")
    
    if not average_df.empty:
        average_df.to_csv(output_dir / 'average_detailed.csv', index=False)
        print(f"Saved average data with shape {average_df.shape} to CSV")
    else:
        print("No average data to save")
        # Create a minimal CSV with headers
        with open(output_dir / 'average_detailed.csv', 'w') as f:
            f.write("Baseline,Baseline + RAG,Fine-tuned,Fine-tuned + RAG\n")

def main():
    """Main function to create visualization dashboard."""
    parser = argparse.ArgumentParser(description='Generate visualization dashboard for evaluation results')
    parser.add_argument('--results', type=str, default='./results/evaluation_results.json',
                        help='Path to evaluation results JSON file')
    parser.add_argument('--output', type=str, default='./results/dashboard',
                        help='Output directory for dashboard')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results(args.results)
    
    # Create summary dataframe
    summary_df = create_summary_dataframe(results)
    
    # Create detailed dataframes
    factuality_df, relevance_df, average_df = create_detailed_dataframes(results)
    
    # Plot summary bar chart
    plot_summary_bar_chart(summary_df, output_dir)
    
    # Plot detailed boxplots
    plot_detailed_boxplots(factuality_df, relevance_df, average_df, output_dir)
    
    # Plot score distributions
    plot_score_distributions(factuality_df, relevance_df, average_df, output_dir)
    
    # Create image paths dictionary
    image_paths = {
        'Model Performance Comparison': output_dir / 'summary_bar_chart.png',
        'Factuality Scores Distribution': output_dir / 'factuality_boxplot.png',
        'Relevance Scores Distribution': output_dir / 'relevance_boxplot.png',
        'Average Scores Distribution': output_dir / 'average_boxplot.png',
        'Factuality Score Distributions': output_dir / 'factuality_distribution.png',
        'Relevance Score Distributions': output_dir / 'relevance_distribution.png',
        'Average Score Distributions': output_dir / 'average_distribution.png'
    }
    
    # Create HTML report
    html_report = create_html_report(summary_df, output_dir, image_paths)
    
    print(f"Dashboard created successfully at {html_report}")
    
    # Save summary as CSV
    summary_df.to_csv(output_dir / 'summary.csv', index=False)
    
    # Save detailed results as CSV
    save_detailed_data(factuality_df, relevance_df, average_df, output_dir)

if __name__ == '__main__':
    main()
