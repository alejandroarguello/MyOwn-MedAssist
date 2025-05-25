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
    detailed = results['detailed']
    
    # Find the minimum length across all arrays to ensure equal length
    min_length = float('inf')
    for model in ['baseline', 'baseline_rag', 'fine_tuned', 'fine_tuned_rag']:
        if model in detailed:
            for metric in ['factuality', 'relevance']:
                if metric in detailed[model]:
                    min_length = min(min_length, len(detailed[model][metric]))
    
    if min_length == float('inf'):
        min_length = 0
    
    # Create dataframes for each metric with equal length arrays
    factuality_data = {}
    relevance_data = {}
    
    for model, display_name in [
        ('baseline', 'Baseline'),
        ('baseline_rag', 'Baseline + RAG'),
        ('fine_tuned', 'Fine-tuned'),
        ('fine_tuned_rag', 'Fine-tuned + RAG')
    ]:
        if model in detailed:
            # Get factuality data and ensure it's the right length
            if 'factuality' in detailed[model]:
                factuality_data[display_name] = detailed[model]['factuality'][:min_length]
            else:
                factuality_data[display_name] = [0] * min_length
                
            # Get relevance data and ensure it's the right length
            if 'relevance' in detailed[model]:
                relevance_data[display_name] = detailed[model]['relevance'][:min_length]
            else:
                relevance_data[display_name] = [0] * min_length
    
    factuality_df = pd.DataFrame(factuality_data)
    relevance_df = pd.DataFrame(relevance_data)
    
    # Calculate average for each example
    average_data = {}
    for model, display_name in [
        ('baseline', 'Baseline'),
        ('baseline_rag', 'Baseline + RAG'),
        ('fine_tuned', 'Fine-tuned'),
        ('fine_tuned_rag', 'Fine-tuned + RAG')
    ]:
        if model in detailed:
            factuality = factuality_data[display_name]
            relevance = relevance_data[display_name]
            
            # Calculate average
            average = [(f + r) / 2 for f, r in zip(factuality, relevance)]
            average_data[display_name] = average
    
    average_df = pd.DataFrame(average_data)
    
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
    # Factuality boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=factuality_df)
    plt.title('Factuality Scores Distribution')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'factuality_boxplot.png')
    plt.close()
    
    # Relevance boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=relevance_df)
    plt.title('Relevance Scores Distribution')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'relevance_boxplot.png')
    plt.close()
    
    # Average boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=average_df)
    plt.title('Average Scores Distribution')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'average_boxplot.png')
    plt.close()

def plot_score_distributions(factuality_df, relevance_df, average_df, output_dir):
    """Create histograms for score distributions."""
    # Factuality histograms
    plt.figure(figsize=(12, 8))
    for column in factuality_df.columns:
        sns.kdeplot(factuality_df[column], label=column)
    plt.title('Factuality Score Distributions')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'factuality_distribution.png')
    plt.close()
    
    # Relevance histograms
    plt.figure(figsize=(12, 8))
    for column in relevance_df.columns:
        sns.kdeplot(relevance_df[column], label=column)
    plt.title('Relevance Score Distributions')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'relevance_distribution.png')
    plt.close()
    
    # Average histograms
    plt.figure(figsize=(12, 8))
    for column in average_df.columns:
        sns.kdeplot(average_df[column], label=column)
    plt.title('Average Score Distributions')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'average_distribution.png')
    plt.close()

def create_html_report(summary_df, output_dir, image_paths):
    """Create an HTML report with all visualizations."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical QA Evaluation Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .visualization {{
                margin: 20px 0;
                text-align: center;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .summary {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .highlight {{
                font-weight: bold;
                color: #3498db;
            }}
        </style>
    </head>
    <body>
        <h1>Medical QA Evaluation Results</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>This report presents the evaluation results for different medical QA model configurations.</p>
        </div>
        
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Factuality</th>
                <th>Relevance</th>
                <th>Average</th>
            </tr>
    """
    
    # Add rows for each model
    for _, row in summary_df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['Model']}</td>
                <td>{row['Factuality']:.4f} ± {row['Factuality Std']:.4f}</td>
                <td>{row['Relevance']:.4f} ± {row['Relevance Std']:.4f}</td>
                <td>{row['Average']:.4f} ± {row['Average Std']:.4f}</td>
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
            <h3>{title}</h3>
            <img src="{path.name}" alt="{title}">
        </div>
        """
    
    # Add insights
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

def main():
    """Main function to create visualization dashboard."""
    parser = argparse.ArgumentParser(description='Generate visualization dashboard for evaluation results')
    parser.add_argument('--results', type=str, default='../results/evaluation_results.json',
                        help='Path to evaluation results JSON file')
    parser.add_argument('--output', type=str, default='../results/dashboard',
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
    factuality_df.to_csv(output_dir / 'factuality_detailed.csv', index=False)
    relevance_df.to_csv(output_dir / 'relevance_detailed.csv', index=False)
    average_df.to_csv(output_dir / 'average_detailed.csv', index=False)

if __name__ == '__main__':
    main()
