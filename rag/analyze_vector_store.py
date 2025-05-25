#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze and visualize vector embeddings:
1. Generate synthetic embeddings for demonstration
2. Perform dimensionality reduction for visualization
3. Generate insights and statistics
4. Create visualizations
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import List, Dict, Any, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import umap
import jsonlines
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PUBMED_DATA_PATH = Path("../data/raw/pubmed/pubmed_abstracts.jsonl")
OUTPUT_DIR = Path("./analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_pubmed_data() -> List[Dict]:
    """Load PubMed abstracts data.
    
    Returns:
        List of dictionaries containing PubMed abstract data
    """
    logger.info(f"Loading PubMed abstracts from {PUBMED_DATA_PATH}")
    
    if not PUBMED_DATA_PATH.exists():
        logger.warning(f"PubMed abstracts file not found at {PUBMED_DATA_PATH}")
        return []
    
    try:
        with jsonlines.open(PUBMED_DATA_PATH) as reader:
            abstracts = list(reader)
        logger.info(f"Loaded {len(abstracts)} PubMed abstracts")
        return abstracts
    except Exception as e:
        logger.error(f"Error loading PubMed abstracts: {e}")
        return []

def generate_synthetic_embeddings(num_samples: int = 1000, dimension: int = 1536) -> Tuple[np.ndarray, List[Dict]]:
    """Generate synthetic embeddings for visualization purposes.
    
    Args:
        num_samples: Number of synthetic embeddings to generate
        dimension: Dimension of the embeddings
        
    Returns:
        Tuple of (embeddings array, metadata list)
    """
    logger.info(f"Generating {num_samples} synthetic embeddings with dimension {dimension}")
    
    # Load real data for metadata
    pubmed_data = load_pubmed_data()
    
    # Determine number of clusters/topics
    num_clusters = 5
    samples_per_cluster = num_samples // num_clusters
    
    # Generate synthetic embeddings with cluster structure
    embeddings = np.zeros((num_samples, dimension), dtype=np.float32)
    metadata = []
    
    # Create cluster centers
    cluster_centers = np.random.randn(num_clusters, dimension)
    
    # Generate samples around cluster centers
    for i in range(num_clusters):
        start_idx = i * samples_per_cluster
        end_idx = start_idx + samples_per_cluster if i < num_clusters - 1 else num_samples
        
        # Generate points around cluster center
        cluster_samples = np.random.randn(end_idx - start_idx, dimension) * 0.1 + cluster_centers[i]
        embeddings[start_idx:end_idx] = cluster_samples
        
        # Normalize embeddings to unit length (cosine similarity)
        norms = np.linalg.norm(embeddings[start_idx:end_idx], axis=1, keepdims=True)
        embeddings[start_idx:end_idx] /= norms
        
        # Create metadata
        for j in range(start_idx, end_idx):
            # Use real metadata if available, otherwise synthetic
            if pubmed_data and j < len(pubmed_data):
                abstract = pubmed_data[j]
                metadata.append({
                    "id": f"doc_{j}",
                    "title": abstract.get("title", f"Document {j}"),
                    "text": abstract.get("abstract", f"Abstract {j}"),
                    "source": "PubMed",
                    "cluster": i,
                    "query_term": abstract.get("query_term", "unknown")
                })
            else:
                # Generate synthetic metadata
                metadata.append({
                    "id": f"doc_{j}",
                    "title": f"Document {j}",
                    "text": f"This is a synthetic document in cluster {i}",
                    "source": "PubMed",
                    "cluster": i,
                    "query_term": ["diabetes", "hypertension", "asthma", "cancer", "COVID-19"][i % 5]
                })
    
    logger.info(f"Generated {len(embeddings)} synthetic embeddings with {len(metadata)} metadata entries")
    return embeddings, metadata

def reduce_dimensions(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """Reduce dimensionality of embeddings for visualization.
    
    Args:
        embeddings: High-dimensional embeddings
        
    Returns:
        Dictionary with reduced embeddings using different techniques
    """
    logger.info("Reducing dimensionality of embeddings")
    
    reduced_embeddings = {}
    
    # 1. PCA (fast, linear)
    logger.info("Performing PCA...")
    pca = PCA(n_components=3)
    reduced_embeddings['pca'] = pca.fit_transform(embeddings)
    
    # 2. t-SNE (better for visualization, non-linear, slower)
    logger.info("Performing t-SNE...")
    tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
    reduced_embeddings['tsne'] = tsne.fit_transform(embeddings)
    
    # 3. UMAP (better preserves global structure, faster than t-SNE)
    logger.info("Performing UMAP...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced_embeddings['umap'] = reducer.fit_transform(embeddings)
    
    return reduced_embeddings

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Cluster embeddings using KMeans.
    
    Args:
        embeddings: High-dimensional embeddings
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels for each embedding
    """
    logger.info(f"Clustering embeddings into {n_clusters} clusters")
    
    # Use KMeans for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels

def calculate_statistics(embeddings: np.ndarray, nodes: List[Dict]) -> Dict:
    """Calculate statistics about the embeddings and nodes.
    
    Args:
        embeddings: High-dimensional embeddings
        nodes: List of node metadata
        
    Returns:
        Dictionary with statistics
    """
    logger.info("Calculating statistics")
    
    stats = {}
    
    # Basic stats
    stats['num_nodes'] = len(nodes)
    stats['embedding_dimension'] = embeddings.shape[1]
    
    # Source distribution
    source_counts = {}
    for node in nodes:
        source = node.get('metadata', {}).get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    stats['source_distribution'] = source_counts
    
    # Embedding statistics
    stats['embedding_norm_mean'] = float(np.mean(np.linalg.norm(embeddings, axis=1)))
    stats['embedding_norm_std'] = float(np.std(np.linalg.norm(embeddings, axis=1)))
    
    # Calculate average pairwise cosine similarity
    # (Sample to avoid memory issues with large datasets)
    if len(embeddings) > 1000:
        sample_indices = np.random.choice(len(embeddings), 1000, replace=False)
        sample_embeddings = embeddings[sample_indices]
    else:
        sample_embeddings = embeddings
    
    # Normalize for cosine similarity
    normalized_embeddings = sample_embeddings / np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # Mask the diagonal (self-similarity)
    np.fill_diagonal(similarity_matrix, 0)
    
    stats['avg_cosine_similarity'] = float(similarity_matrix.mean())
    stats['max_cosine_similarity'] = float(similarity_matrix.max())
    
    return stats

def extract_cluster_topics(nodes: List[Dict], cluster_labels: np.ndarray, n_topics: int = 10) -> Dict[int, Dict]:
    """Extract the most common terms in each cluster to identify topics.
    
    Args:
        nodes: List of node metadata
        cluster_labels: Cluster labels for each embedding
        n_topics: Number of top terms to extract per cluster
        
    Returns:
        Dictionary mapping cluster IDs to topic information
    """
    logger.info("Extracting cluster topics")
    
    # Group documents by cluster
    cluster_docs = {}
    for i, node in enumerate(nodes):
        cluster = int(cluster_labels[i])
        if cluster not in cluster_docs:
            cluster_docs[cluster] = []
        
        # Get text from node (title and content)
        text = ""
        if "title" in node and node["title"]:
            text += node["title"] + " "
        if "text" in node and node["text"]:
            text += node["text"]
        if "query_term" in node and node["query_term"]:
            query_term = node["query_term"]
            if isinstance(query_term, list):
                text += " " + " ".join(query_term)
            else:
                text += " " + query_term
        
        cluster_docs[cluster].append(text)
    
    # Extract top terms for each cluster using TF-IDF
    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=2,
        stop_words='english',
        ngram_range=(1, 2)  # Include bigrams
    )
    
    # Combine all documents for vectorization
    all_docs = []
    cluster_indices = []
    for cluster, docs in cluster_docs.items():
        all_docs.extend(docs)
        cluster_indices.extend([cluster] * len(docs))
    
    # Vectorize if we have documents
    if not all_docs:
        return {}
    
    try:
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top terms for each cluster
        cluster_topics = {}
        for cluster in sorted(cluster_docs.keys()):
            # Get indices of documents in this cluster
            indices = [i for i, c in enumerate(cluster_indices) if c == cluster]
            if not indices:
                continue
            
            # Get the TF-IDF scores for documents in this cluster
            cluster_tfidf = tfidf_matrix[indices]
            
            # Sum TF-IDF scores across documents in the cluster
            cluster_scores = np.asarray(cluster_tfidf.sum(axis=0)).flatten()
            
            # Get top terms
            top_indices = cluster_scores.argsort()[::-1][:n_topics]
            top_terms = [(feature_names[i], float(cluster_scores[i])) for i in top_indices]
            
            # Get query terms associated with this cluster
            query_terms = []
            for i in indices:
                node = nodes[i]
                if "query_term" in node and node["query_term"]:
                    query_term = node["query_term"]
                    if isinstance(query_term, list):
                        query_terms.extend(query_term)
                    else:
                        query_terms.append(query_term)
            
            # Count query terms
            query_term_counts = Counter(query_terms)
            top_query_terms = query_term_counts.most_common(3)
            
            # Store cluster information
            cluster_topics[cluster] = {
                "top_terms": top_terms,
                "document_count": len(indices),
                "top_query_terms": top_query_terms
            }
        
        return cluster_topics
    except Exception as e:
        logger.error(f"Error extracting cluster topics: {e}")
        return {}

def create_cluster_wordclouds(nodes: List[Dict], cluster_labels: np.ndarray):
    """Create word clouds for each cluster.
    
    Args:
        nodes: List of node metadata
        cluster_labels: Cluster labels for each embedding
    """
    logger.info("Creating cluster word clouds")
    
    # Group documents by cluster
    cluster_docs = {}
    for i, node in enumerate(nodes):
        cluster = int(cluster_labels[i])
        if cluster not in cluster_docs:
            cluster_docs[cluster] = []
        
        # Get text from node
        text = ""
        if "title" in node and node["title"]:
            text += node["title"] + " "
        if "text" in node and node["text"]:
            text += node["text"]
        
        cluster_docs[cluster].append(text)
    
    # Create word cloud for each cluster
    for cluster, docs in cluster_docs.items():
        if not docs:
            continue
        
        # Combine all documents in the cluster
        text = " ".join(docs)
        
        # Create word cloud
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                contour_width=3,
                contour_color="steelblue"
            ).generate(text)
            
            # Save word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Cluster {cluster} Word Cloud")
            plt.tight_layout(pad=0)
            plt.savefig(OUTPUT_DIR / f"cluster_{cluster}_wordcloud.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.error(f"Error creating word cloud for cluster {cluster}: {e}")

def create_visualizations(reduced_embeddings: Dict[str, np.ndarray], 
                          nodes: List[Dict], 
                          cluster_labels: np.ndarray,
                          stats: Dict):
    """Create visualizations of the embeddings.
    
    Args:
        reduced_embeddings: Dictionary with reduced embeddings
        nodes: List of node metadata
        cluster_labels: Cluster labels for each embedding
        stats: Statistics dictionary
    """
    logger.info("Creating visualizations")
    
    # Extract cluster topics
    cluster_topics = extract_cluster_topics(nodes, cluster_labels)
    
    # Create word clouds for each cluster
    create_cluster_wordclouds(nodes, cluster_labels)
    
    # Save cluster topics to file
    with open(OUTPUT_DIR / "cluster_topics.json", "w") as f:
        json.dump(cluster_topics, f, indent=2)
    
    # Extract metadata for coloring and hover information
    sources = [node.get('source', 'unknown') for node in nodes]
    titles = [node.get('title', f"Document {i}") for i, node in enumerate(nodes)]
    query_terms = [node.get('query_term', 'unknown') for node in nodes]
    
    # Create a DataFrame for easier plotting
    df_base = pd.DataFrame({
        'cluster': cluster_labels,
        'source': sources,
        'title': titles,
        'query_term': query_terms
    })
    
    # Add cluster topic information
    cluster_names = {}
    for cluster, info in cluster_topics.items():
        if info["top_terms"]:
            top_terms = [term for term, score in info["top_terms"][:3]]
            cluster_names[cluster] = f"Cluster {cluster}: {', '.join(top_terms)}"
        else:
            cluster_names[cluster] = f"Cluster {cluster}"
    
    df_base['cluster_name'] = [cluster_names.get(c, f"Cluster {c}") for c in df_base['cluster']]
    
    # 1. Create 3D scatter plots for each dimensionality reduction technique
    for method, embeddings in reduced_embeddings.items():
        logger.info(f"Creating 3D scatter plot for {method}")
        
        df = df_base.copy()
        df['x'] = embeddings[:, 0]
        df['y'] = embeddings[:, 1]
        df['z'] = embeddings[:, 2]
        
        # Plot by source with hover information
        fig = px.scatter_3d(
            df, x='x', y='y', z='z', 
            color='source',
            hover_name='title',
            hover_data={
                'query_term': True,
                'cluster_name': True,
                'x': False,
                'y': False,
                'z': False
            },
            title=f'Vector Embeddings - {method.upper()} (Colored by Source)'
        )
        fig.update_layout(scene=dict(aspectmode='cube'))
        fig.write_html(OUTPUT_DIR / f'3d_scatter_{method}_by_source.html')
        
        # Plot by cluster with hover information
        fig = px.scatter_3d(
            df, x='x', y='y', z='z', 
            color='cluster_name',
            hover_name='title',
            hover_data={
                'query_term': True,
                'source': True,
                'x': False,
                'y': False,
                'z': False
            },
            title=f'Vector Embeddings - {method.upper()} (Colored by Cluster)'
        )
        fig.update_layout(scene=dict(aspectmode='cube'))
        fig.write_html(OUTPUT_DIR / f'3d_scatter_{method}_by_cluster.html')
    
    # 2. Create source distribution pie chart
    source_dist = stats['source_distribution']
    fig = px.pie(values=list(source_dist.values()), names=list(source_dist.keys()),
                title='Distribution of Document Sources')
    fig.write_html(OUTPUT_DIR / 'source_distribution.html')
    
    # 3. Create cluster analysis with topic information
    cluster_info = []
    for cluster, info in cluster_topics.items():
        if info["top_terms"]:
            top_terms = ", ".join([term for term, score in info["top_terms"][:5]])
        else:
            top_terms = "No significant terms"
            
        cluster_info.append({
            "cluster": cluster,
            "document_count": info["document_count"],
            "top_terms": top_terms
        })
    
    cluster_info_df = pd.DataFrame(cluster_info)
    if not cluster_info_df.empty:
        cluster_info_df = cluster_info_df.sort_values("cluster")
        
        fig = px.bar(
            cluster_info_df, 
            x="cluster", 
            y="document_count",
            hover_data=["top_terms"],
            labels={
                'cluster': 'Cluster', 
                'document_count': 'Number of Documents',
                'top_terms': 'Top Terms'
            },
            title='Document Distribution by Cluster with Top Terms'
        )
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        fig.write_html(OUTPUT_DIR / 'cluster_distribution.html')
    else:
        # Fallback to simple cluster counts
        cluster_counts = df_base['cluster'].value_counts().sort_index()
        fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Number of Documents'},
                    title='Document Distribution by Cluster')
        fig.write_html(OUTPUT_DIR / 'cluster_distribution.html')
    
    # 4. Create cross-tabulation of clusters vs sources
    cross_tab = pd.crosstab(df_base['cluster_name'], df_base['source'])
    fig = px.imshow(cross_tab, text_auto=True,
                   labels=dict(x="Source", y="Cluster", color="Count"),
                   title='Cluster vs Source Distribution')
    fig.write_html(OUTPUT_DIR / 'cluster_source_distribution.html')
    
    # 5. Create cluster topic summary table
    cluster_summary = []
    for cluster, info in cluster_topics.items():
        if info["top_terms"]:
            top_terms = "<br>".join([f"{term} ({score:.3f})" for term, score in info["top_terms"][:10]])
        else:
            top_terms = "No significant terms"
            
        if info["top_query_terms"]:
            query_terms = "<br>".join([f"{term} ({count})" for term, count in info["top_query_terms"]])
        else:
            query_terms = "Unknown"
            
        cluster_summary.append({
            "cluster": cluster,
            "document_count": info["document_count"],
            "top_terms": top_terms,
            "query_terms": query_terms
        })
    
    if cluster_summary:
        cluster_summary_df = pd.DataFrame(cluster_summary).sort_values("cluster")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Cluster", "Document Count", "Top Terms", "Query Terms"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    cluster_summary_df["cluster"],
                    cluster_summary_df["document_count"],
                    cluster_summary_df["top_terms"],
                    cluster_summary_df["query_terms"]
                ],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(title="Cluster Topic Summary")
        fig.write_html(OUTPUT_DIR / 'cluster_topic_summary.html')
    
    # 5. Create statistics summary
    with open(OUTPUT_DIR / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 6. Create similarity heatmap
    # Sample nodes for better visualization
    if len(nodes) > 100:
        sample_indices = np.random.choice(len(nodes), 100, replace=False)
    else:
        sample_indices = np.arange(len(nodes))
    
    # Get embeddings for sampled nodes
    sample_embeddings = np.array([reduced_embeddings['pca'][i] for i in sample_indices])
    
    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(sample_embeddings, 'euclidean'))
    
    # Create heatmap
    fig = px.imshow(distances, 
                   labels=dict(x="Document Index", y="Document Index", color="Distance"),
                   title='Pairwise Distance Heatmap (PCA, Sample of 100 Documents)')
    fig.write_html(OUTPUT_DIR / 'distance_heatmap.html')
    
    logger.info(f"Visualizations saved to {OUTPUT_DIR}")

def analyze_nearest_neighbors(embeddings: np.ndarray, metadata: List[Dict], k: int = 5):
    """Analyze nearest neighbors for each embedding using cosine similarity.
    
    Args:
        embeddings: High-dimensional embeddings
        metadata: List of metadata dictionaries
        k: Number of nearest neighbors to find
    """
    logger.info(f"Analyzing nearest neighbors (k={k})")
    
    # Sample documents for analysis
    sample_size = min(10, len(metadata))
    sample_indices = np.random.choice(len(metadata), sample_size, replace=False)
    
    # Create a report
    report = []
    
    # Normalize embeddings for cosine similarity
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    for idx in sample_indices:
        # Get the query embedding
        query_embedding = normalized_embeddings[idx]
        
        # Calculate cosine similarities
        similarities = np.dot(normalized_embeddings, query_embedding)
        
        # Get indices of top k+1 similar items
        top_indices = np.argsort(similarities)[::-1][:k+1]
        
        # Skip the first result if it's self
        if top_indices[0] == idx:
            top_indices = top_indices[1:k+1]
        else:
            top_indices = top_indices[:k]
        
        # Get the similarities for these indices
        top_similarities = similarities[top_indices]
        
        # Get the query document
        query_doc = metadata[idx]
        
        # Get the neighbor documents
        neighbor_docs = [metadata[i] for i in top_indices]
        
        # Create a report entry
        entry = {
            "query": {
                "id": query_doc["id"],
                "title": query_doc["title"],
                "text": query_doc["text"][:200] + "..." if query_doc["text"] and len(query_doc["text"]) > 200 else query_doc.get("text", ""),
                "source": query_doc.get("source", "unknown"),
                "query_term": query_doc.get("query_term", "unknown")
            },
            "neighbors": []
        }
        
        for i, (neighbor, similarity) in enumerate(zip(neighbor_docs, top_similarities)):
            entry["neighbors"].append({
                "rank": i+1,
                "id": neighbor["id"],
                "title": neighbor["title"],
                "text": neighbor["text"][:200] + "..." if neighbor["text"] and len(neighbor["text"]) > 200 else neighbor.get("text", ""),
                "source": neighbor.get("source", "unknown"),
                "query_term": neighbor.get("query_term", "unknown"),
                "similarity": float(similarity)
            })
        
        report.append(entry)
    
    # Save the report
    with open(OUTPUT_DIR / 'nearest_neighbors_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Nearest neighbors report saved to {OUTPUT_DIR / 'nearest_neighbors_report.json'}")

def main():
    """Main function to analyze vector embeddings."""
    logger.info("Starting vector embedding analysis")
    
    # Generate synthetic embeddings
    embeddings, metadata = generate_synthetic_embeddings(num_samples=1000)
    
    # Reduce dimensions for visualization
    reduced_embeddings = reduce_dimensions(embeddings)
    
    # Cluster embeddings
    cluster_labels = cluster_embeddings(embeddings)
    
    # Calculate statistics
    stats = calculate_statistics(embeddings, metadata)
    
    # Create visualizations
    create_visualizations(reduced_embeddings, metadata, cluster_labels, stats)
    
    # Analyze nearest neighbors
    analyze_nearest_neighbors(embeddings, metadata)
    
    # Create a simple HTML index file to navigate the visualizations
    create_index_html()
    
    logger.info("Vector embedding analysis complete")
    logger.info(f"Results saved to {OUTPUT_DIR}")
    
def create_index_html():
    """Create an HTML index file to navigate all visualizations."""
    html_files = list(OUTPUT_DIR.glob("*.html"))
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vector Embedding Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            .section { margin-bottom: 30px; }
            .visualization { margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Vector Embedding Analysis</h1>
        
        <div class="section">
            <h2>Cluster Analysis</h2>
    """
    
    # Add cluster analysis visualizations
    cluster_files = [f for f in html_files if "cluster" in f.name]
    for file in cluster_files:
        html_content += f"""
            <div class="visualization">
                <a href="{file.name}" target="_blank">{file.stem.replace('_', ' ').title()}</a>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>3D Visualizations</h2>
    """
    
    # Add 3D scatter plots
    scatter_files = [f for f in html_files if "scatter" in f.name]
    for file in scatter_files:
        html_content += f"""
            <div class="visualization">
                <a href="{file.name}" target="_blank">{file.stem.replace('_', ' ').title()}</a>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Other Visualizations</h2>
    """
    
    # Add other visualizations
    other_files = [f for f in html_files if "cluster" not in f.name and "scatter" not in f.name]
    for file in other_files:
        html_content += f"""
            <div class="visualization">
                <a href="{file.name}" target="_blank">{file.stem.replace('_', ' ').title()}</a>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write the index file
    with open(OUTPUT_DIR / "index.html", "w") as f:
        f.write(html_content)
    
    logger.info(f"Created index.html in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
