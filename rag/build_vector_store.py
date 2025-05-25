#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build a FAISS vector store for RAG using medical data sources:
1. Fetch data from PubMed, clinical guidelines, and drug information
2. Process and chunk documents appropriately
3. Create embeddings
4. Build and save FAISS index
"""

import os
import json
import jsonlines
import logging
import requests
import time
import io
import tarfile
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm

# LlamaIndex imports
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

# Import faiss
import faiss

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
PROCESSED_DATA_DIR = Path("../data/processed")
RAW_DATA_DIR = Path("../data/raw")
FAISS_STORE_DIR = Path("./faiss_store")
EMBED_MODEL = "text-embedding-3-small"  # OpenAI embedding model

# PubMed constants
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_SEARCH_URL = f"{PUBMED_BASE_URL}/esearch.fcgi"
PUBMED_FETCH_URL = f"{PUBMED_BASE_URL}/efetch.fcgi"
PUBMED_SUMMARY_URL = f"{PUBMED_BASE_URL}/esummary.fcgi"
PMC_OA_FTP_BASE = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk"

# NICE API constants
NICE_API_BASE = "https://api.nice.org.uk/services/v1"
NICE_GUIDANCE_ENDPOINT = f"{NICE_API_BASE}/guidance"

# DailyMed constants
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
DAILYMED_SPLS_URL = f"{DAILYMED_BASE_URL}/spls.json"

# Data directories
PUBMED_DATA_DIR = RAW_DATA_DIR / "pubmed"
GUIDELINES_DATA_DIR = RAW_DATA_DIR / "guidelines"
DRUG_DATA_DIR = RAW_DATA_DIR / "drugs"

# Create directories if they don't exist
for directory in [PUBMED_DATA_DIR, GUIDELINES_DATA_DIR, DRUG_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def fetch_pubmed_abstracts(query_terms: List[str], max_results: int = 1000) -> List[Dict]:
    """Fetch PubMed abstracts using NCBI E-utilities.
    
    Args:
        query_terms: List of search terms
        max_results: Maximum number of results to fetch
        
    Returns:
        List of dictionaries containing PubMed abstract data
    """
    logger.info(f"Fetching PubMed abstracts for terms: {query_terms}")
    
    abstracts = []
    
    for term in query_terms:
        # Step 1: Search for PMIDs
        search_params = {
            "db": "pubmed",
            "term": term,
            "retmax": min(max_results, 1000),  # API limit
            "retmode": "json",
            "usehistory": "y"
        }
        
        try:
            search_response = requests.get(PUBMED_SEARCH_URL, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            pmids = search_data["esearchresult"]["idlist"]
            webenv = search_data["esearchresult"]["webenv"]
            query_key = search_data["esearchresult"]["querykey"]
            
            logger.info(f"Found {len(pmids)} PMIDs for term: {term}")
            
            # Step 2: Fetch abstracts in batches
            batch_size = 100
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]
                
                fetch_params = {
                    "db": "pubmed",
                    "query_key": query_key,
                    "WebEnv": webenv,
                    "retstart": i,
                    "retmax": batch_size,
                    "retmode": "xml",
                    "rettype": "abstract"
                }
                
                fetch_response = requests.get(PUBMED_FETCH_URL, params=fetch_params)
                fetch_response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(fetch_response.text)
                articles = root.findall(".//PubmedArticle")
                
                for article in articles:
                    pmid_elem = article.find(".//PMID")
                    title_elem = article.find(".//ArticleTitle")
                    abstract_elem = article.find(".//AbstractText")
                    
                    if pmid_elem is not None and (title_elem is not None or abstract_elem is not None):
                        pmid = pmid_elem.text
                        title = title_elem.text if title_elem is not None else ""
                        abstract = abstract_elem.text if abstract_elem is not None else ""
                        
                        if title or abstract:
                            abstracts.append({
                                "pmid": pmid,
                                "title": title,
                                "abstract": abstract,
                                "source": "PubMed",
                                "query_term": term
                            })
                
                # Respect API rate limits
                time.sleep(0.34)  # Max 3 requests per second
                
        except Exception as e:
            logger.error(f"Error fetching PubMed abstracts for term {term}: {e}")
    
    # Save abstracts to file
    abstracts_file = PUBMED_DATA_DIR / "pubmed_abstracts.jsonl"
    with jsonlines.open(abstracts_file, mode='w') as writer:
        writer.write_all(abstracts)
    
    logger.info(f"Saved {len(abstracts)} PubMed abstracts to {abstracts_file}")
    
    return abstracts

def fetch_nice_guidelines(page_size: int = 50, max_pages: int = 10) -> List[Dict]:
    """Fetch clinical guidelines from NICE API.
    
    Args:
        page_size: Number of results per page
        max_pages: Maximum number of pages to fetch
        
    Returns:
        List of dictionaries containing guideline data
    """
    logger.info("Fetching clinical guidelines from NICE API")
    
    guidelines = []
    
    # You would need to request an API key for production use
    # For this prototype, we'll use the public API with limited rate
    
    try:
        for page in range(1, max_pages + 1):
            params = {
                "page": page,
                "pageSize": page_size
            }
            
            response = requests.get(NICE_GUIDANCE_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("guidance", []):
                guideline_id = item.get("id")
                title = item.get("title")
                
                # Fetch detailed guideline content
                if guideline_id:
                    guideline_url = f"{NICE_GUIDANCE_ENDPOINT}/{guideline_id}"
                    detail_response = requests.get(guideline_url)
                    
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        
                        # Extract sections and recommendations
                        sections = []
                        if "sections" in detail_data:
                            for section in detail_data["sections"]:
                                section_title = section.get("title", "")
                                section_content = section.get("content", "")
                                sections.append(f"{section_title}\n{section_content}")
                        
                        guidelines.append({
                            "id": guideline_id,
                            "title": title,
                            "sections": sections,
                            "source": "NICE Guidelines"
                        })
                
                # Respect API rate limits
                time.sleep(1)  # Be conservative with external API
            
            # Check if we've reached the last page
            if len(data.get("guidance", [])) < page_size:
                break
    
    except Exception as e:
        logger.error(f"Error fetching NICE guidelines: {e}")
    
    # Save guidelines to file
    guidelines_file = GUIDELINES_DATA_DIR / "nice_guidelines.jsonl"
    with jsonlines.open(guidelines_file, mode='w') as writer:
        writer.write_all(guidelines)
    
    logger.info(f"Saved {len(guidelines)} NICE guidelines to {guidelines_file}")
    
    return guidelines

def fetch_dailymed_data(limit: int = 100) -> List[Dict]:
    """Fetch drug information from DailyMed API.
    
    Args:
        limit: Maximum number of drug labels to fetch
        
    Returns:
        List of dictionaries containing drug information
    """
    logger.info("Fetching drug information from DailyMed API")
    
    drug_data = []
    
    try:
        # Get list of available SPLs (Structured Product Labels)
        params = {
            "pagesize": min(limit, 100),  # API limit
            "format": "json"
        }
        
        response = requests.get(DAILYMED_SPLS_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get("data", []):
            set_id = item.get("setid")
            
            if set_id:
                # Fetch detailed drug information
                detail_url = f"{DAILYMED_BASE_URL}/spls/{set_id}.json"
                detail_response = requests.get(detail_url)
                
                if detail_response.status_code == 200:
                    detail_data = detail_response.json()
                    
                    # Extract relevant sections
                    sections = []
                    if "data" in detail_data and "sections" in detail_data["data"]:
                        for section in detail_data["data"]["sections"]:
                            section_name = section.get("name", "")
                            section_text = section.get("content", "")
                            sections.append(f"{section_name}\n{section_text}")
                    
                    drug_data.append({
                        "set_id": set_id,
                        "name": item.get("title", ""),
                        "sections": sections,
                        "source": "DailyMed"
                    })
                
                # Respect API rate limits
                time.sleep(0.5)
    
    except Exception as e:
        logger.error(f"Error fetching DailyMed data: {e}")
    
    # Save drug data to file
    drugs_file = DRUG_DATA_DIR / "dailymed_drugs.jsonl"
    with jsonlines.open(drugs_file, mode='w') as writer:
        writer.write_all(drug_data)
    
    logger.info(f"Saved {len(drug_data)} drug labels to {drugs_file}")
    
    return drug_data

def load_medical_data(fetch_new: bool = False) -> List[Document]:
    """Load medical data from various sources for indexing.
    
    Args:
        fetch_new: Whether to fetch new data or use cached data
        
    Returns:
        List of Document objects ready for indexing
    """
    logger.info("Loading medical data for indexing")
    
    documents = []
    
    # 1. PubMed abstracts
    pubmed_file = PUBMED_DATA_DIR / "pubmed_abstracts.jsonl"
    if fetch_new or not pubmed_file.exists():
        logger.info("Fetching new PubMed abstracts")
        # Common medical search terms
        search_terms = [
            "diabetes treatment guidelines",
            "hypertension management",
            "asthma therapy",
            "cancer immunotherapy",
            "COVID-19 treatment"
        ]
        abstracts = fetch_pubmed_abstracts(search_terms, max_results=200)
    else:
        logger.info("Loading cached PubMed abstracts")
        abstracts = []
        if pubmed_file.exists():
            with jsonlines.open(pubmed_file) as reader:
                abstracts = list(reader)
    
    for abstract in abstracts:
        title = abstract.get("title", "")
        abstract_text = abstract.get("abstract", "")
        pmid = abstract.get("pmid", "")
        
        if title and abstract_text:
            doc_text = f"Title: {title}\n\nAbstract: {abstract_text}"
            doc = Document(
                text=doc_text,
                metadata={
                    "source": "PubMed",
                    "pmid": pmid,
                    "title": title
                }
            )
            documents.append(doc)
    
    # 2. Clinical guidelines
    guidelines_file = GUIDELINES_DATA_DIR / "nice_guidelines.jsonl"
    if fetch_new or not guidelines_file.exists():
        logger.info("Fetching new clinical guidelines")
        guidelines = fetch_nice_guidelines()
    else:
        logger.info("Loading cached clinical guidelines")
        guidelines = []
        if guidelines_file.exists():
            with jsonlines.open(guidelines_file) as reader:
                guidelines = list(reader)
    
    for guideline in guidelines:
        title = guideline.get("title", "")
        sections = guideline.get("sections", [])
        guideline_id = guideline.get("id", "")
        
        if title and sections:
            doc_text = f"Title: {title}\n\n" + "\n\n".join(sections)
            doc = Document(
                text=doc_text,
                metadata={
                    "source": "NICE Guidelines",
                    "id": guideline_id,
                    "title": title
                }
            )
            documents.append(doc)
    
    # 3. Drug information
    drugs_file = DRUG_DATA_DIR / "dailymed_drugs.jsonl"
    if fetch_new or not drugs_file.exists():
        logger.info("Fetching new drug information")
        drug_data = fetch_dailymed_data()
    else:
        logger.info("Loading cached drug information")
        drug_data = []
        if drugs_file.exists():
            with jsonlines.open(drugs_file) as reader:
                drug_data = list(reader)
    
    for drug in drug_data:
        name = drug.get("name", "")
        sections = drug.get("sections", [])
        set_id = drug.get("set_id", "")
        
        if name and sections:
            doc_text = f"Drug: {name}\n\n" + "\n\n".join(sections)
            doc = Document(
                text=doc_text,
                metadata={
                    "source": "DailyMed",
                    "set_id": set_id,
                    "name": name
                }
            )
            documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} medical documents for indexing")
    return documents

def build_vector_store(documents):
    """Build FAISS vector store from documents."""
    logger.info("Building FAISS vector store")
    
    # Create node parser for splitting documents with source-specific chunking
    node_parsers = {
        "PubMed": SentenceSplitter(chunk_size=300, chunk_overlap=30),  # One abstract per chunk (~200-300 tokens)
        "NICE Guidelines": SentenceSplitter(chunk_size=600, chunk_overlap=50),  # 400-600 tokens for guidelines
        "DailyMed": SentenceSplitter(chunk_size=500, chunk_overlap=50)  # 300-500 tokens for drug labels
    }
    
    # Default parser for other sources
    default_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    # Create embedding model
    embed_model = OpenAIEmbedding(
        model=EMBED_MODEL,
        embed_batch_size=100
    )
    
    # Create FAISS index
    dimension = 1536  # OpenAI embedding dimension
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    
    # Create FAISS vector store
    vector_store = FaissVectorStore(faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create nodes from documents with source-specific chunking
    logger.info("Parsing documents into nodes with source-specific chunking")
    all_nodes = []
    
    # Group documents by source
    source_docs = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in source_docs:
            source_docs[source] = []
        source_docs[source].append(doc)
    
    # Process each source with appropriate chunking
    for source, docs in source_docs.items():
        parser = node_parsers.get(source, default_parser)
        logger.info(f"Processing {len(docs)} documents from {source} with chunk size {parser.chunk_size}")
        nodes = parser.get_nodes_from_documents(docs)
        all_nodes.extend(nodes)
        logger.info(f"Created {len(nodes)} nodes from {source}")
    
    logger.info(f"Created {len(all_nodes)} total nodes from {len(documents)} documents")
    
    # Create index
    logger.info("Creating vector index")
    index = VectorStoreIndex(
        all_nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    # Save index
    os.makedirs(FAISS_STORE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=str(FAISS_STORE_DIR))

    #Also manually save the FAISS binary index
    faiss.write_index(faiss_index, "faiss_store/index_store.faiss")
    
    logger.info(f"Vector store saved to {FAISS_STORE_DIR}")
    
    return index

def main():
    """Main function to build vector store."""
    logger.info("Starting vector store building")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Build vector store for RAG")
    parser.add_argument("--fetch-new", action="store_true", help="Fetch new data instead of using cached data")
    args = parser.parse_args()
    
    # Load medical data from various sources
    documents = load_medical_data(fetch_new=args.fetch_new)
    
    if not documents:
        logger.warning("No documents found. Please check the data sources.")
        return
    
    # Build vector store
    index = build_vector_store(documents)
    
    logger.info("Vector store building complete")

if __name__ == "__main__":
    main()
