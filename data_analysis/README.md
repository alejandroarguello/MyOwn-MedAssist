# Data Analysis Module for MyOwn-MedAssist

This module provides tools and scripts for analyzing and visualizing the medical datasets used in the MyOwn-MedAssist project. The analysis helps understand the characteristics, distributions, and patterns in the PubMedQA, MedMCQA, and Synthea datasets.

## Directory Structure

```
data_analysis/
├── scripts/
│   ├── analyze_datasets.py         # Main analysis script that generates visualizations
│   ├── analyze_finetuning_dataset.py # Script to analyze fine-tuning dataset composition
│   └── interactive_exploration.py # Interactive exploration script for more detailed analysis
├── visualizations/               # Output directory for generated visualizations
└── README.md                     # This file
```

## Features

1. **Dataset Statistics**: Basic statistics about each dataset, including size, text lengths, and unique characteristics.
2. **Text Analysis**: Analysis of questions and answers, including word counts, medical term extraction, and content patterns.
3. **Visualizations**: Various visualizations including:
   - Word clouds
   - Length distributions
   - Top medical terms
   - Dataset comparisons
   - Text embeddings visualization
4. **Dataset-Specific Analysis**:
   - PubMedQA: Analysis of yes/no/maybe answers and context lengths
   - MedMCQA: Analysis of multiple-choice questions and answer distributions
   - Synthea: Analysis of medical conditions and prescribed medications

## Usage

### Running the Analysis Script

To generate a standard set of visualizations and statistics for all datasets:

```bash
python data_analysis/scripts/analyze_datasets.py
```

> **Note:** Our analysis script is designed to be robust and doesn't rely on complex NLTK dependencies.

This will:
1. Analyze each dataset individually
2. Compare statistics across datasets
3. Generate visualizations in the `data_analysis/visualizations/` directory
4. Create a summary report in `data_analysis/visualizations/analysis_summary.txt`

### Interactive Exploration

For more detailed exploration, you can use the interactive exploration script in a Python session or Jupyter notebook:

```python
# In a Python session or Jupyter notebook
from data_analysis.scripts.interactive_exploration import DatasetExplorer

# Create an explorer instance
explorer = DatasetExplorer()

# Load all datasets
explorer.load_all_datasets()

# Get basic statistics for a dataset
stats = explorer.get_basic_stats('PubMedQA')
print(stats)

# Plot length distributions
explorer.plot_length_distributions('MedMCQA')

# Create word clouds
explorer.plot_wordclouds('Synthea')

# Compare datasets
comparison = explorer.compare_datasets()
print(comparison)

# Dataset-specific analysis
explorer.analyze_pubmedqa_specific()
explorer.analyze_medmcqa_specific()
explorer.analyze_synthea_specific()

# Sample examples from a dataset
samples = explorer.sample_examples('PubMedQA', n=3)
```

## Required Dependencies

The analysis scripts require the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- wordcloud
- jsonlines
- scikit-learn
- plotly (for interactive visualizations)

You can install these dependencies using:

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud jsonlines scikit-learn plotly
```

## Visualization Types

1. **Length Distributions**: Histograms showing the distribution of word counts in questions and answers.
2. **Word Clouds**: Visual representations of the most frequent words in questions and answers.
3. **Top Medical Terms**: Bar charts showing the most common medical terms in each dataset.
4. **Dataset Comparison**: Comparative visualizations of dataset sizes and text lengths.
5. **Text Embeddings**: Dimensionality reduction visualizations (PCA, t-SNE) of text embeddings.

## Output Files

The analysis script generates the following files in the `visualizations/` directory:

- `dataset_size_comparison.png`: Comparison of dataset sizes
- `text_length_comparison.png`: Comparison of average text lengths
- `dataset_comparison_summary.csv`: CSV file with comparative statistics
- `analysis_summary.txt`: Text summary of all analysis results
- Dataset-specific visualizations for each dataset (length distributions, word clouds, etc.)

## Extending the Analysis

To add new analysis features:
1. Extend the `analyze_datasets.py` script with new functions
2. Add new methods to the `DatasetExplorer` class in `interactive_exploration.py`
