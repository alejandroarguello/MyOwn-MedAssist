#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download the required datasets:
1. PubMedQA
2. MedMCQA
3. Synthea EHR
"""

import os
import requests
import zipfile
import tarfile
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = Path("../raw")
PUBMEDQA_URL = "https://github.com/pubmedqa/pubmedqa/archive/refs/heads/master.zip"
MEDMCQA_URL = "https://github.com/medmcqa/medmcqa/archive/refs/heads/main.zip"
# Synthea is a bit different - we'll use their GitHub repo to generate synthetic data
SYNTHEA_URL = "https://github.com/synthetichealth/synthea/archive/refs/heads/master.zip"

# Try to import additional libraries for Hugging Face dataset download
try:
    import pandas as pd
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    logger.warning("datasets or pandas library not available. Will download MedMCQA from GitHub only.")
    HUGGINGFACE_AVAILABLE = False

def download_file(url, save_path):
    """Download a file from a URL to a specified path."""
    logger.info(f"Downloading {url} to {save_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Download complete: {save_path}")
    return save_path

def extract_zip(zip_path, extract_to):
    """Extract a zip file to a specified directory."""
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Extraction complete: {extract_to}")

def download_pubmedqa():
    """Download and extract PubMedQA dataset."""
    zip_path = RAW_DATA_DIR / "pubmedqa.zip"
    extract_to = RAW_DATA_DIR / "pubmedqa"
    
    os.makedirs(extract_to, exist_ok=True)
    
    download_file(PUBMEDQA_URL, zip_path)
    extract_zip(zip_path, extract_to)
    
    logger.info("PubMedQA dataset downloaded and extracted")

def download_medmcqa():
    """Download MedMCQA dataset from Hugging Face."""
    extract_to = RAW_DATA_DIR / "medmcqa"
    
    # Create directories
    os.makedirs(extract_to, exist_ok=True)
    
    if HUGGINGFACE_AVAILABLE:
        try:
            logger.info("Downloading MedMCQA dataset from Hugging Face")
            
            # Create CSV directory
            csv_dir = extract_to / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            
            # Load dataset from Hugging Face
            dataset = load_dataset("openlifescienceai/medmcqa")
            logger.info(f"Downloaded dataset with splits: {dataset.keys()}")
            
            # Save each split as a CSV file
            for split_name, split_data in dataset.items():
                output_file = csv_dir / f"{split_name}.csv"
                
                # Convert to pandas DataFrame and save as CSV
                df = pd.DataFrame(split_data)
                df.to_csv(output_file, index=False)
                
                logger.info(f"Saved {len(df)} examples to {output_file}")
            
            logger.info("MedMCQA dataset downloaded from Hugging Face and saved as CSV")
        except Exception as e:
            logger.error(f"Error downloading MedMCQA from Hugging Face: {e}")
            logger.error("MedMCQA dataset download failed. Please install the required libraries and try again.")
            raise e
    else:
        error_msg = "Hugging Face datasets library not available. Cannot download MedMCQA dataset."
        logger.error(error_msg)
        logger.error("Please install the required libraries with: pip install datasets pandas")
        logger.error("Then run this script again.")
        raise ImportError(error_msg)
    
    logger.info("MedMCQA download complete")

def download_synthea():
    """Download and extract Synthea for generating synthetic EHR data."""
    zip_path = RAW_DATA_DIR / "synthea.zip"
    extract_to = RAW_DATA_DIR / "synthea"
    
    os.makedirs(extract_to, exist_ok=True)
    
    download_file(SYNTHEA_URL, zip_path)
    extract_zip(zip_path, extract_to)
    
    logger.info("Synthea downloaded and extracted")
    
    # Automatically generate Synthea data
    logger.info("Generating Synthea synthetic EHR data...")
    
    # Get the path to the generate_synthea_data.sh script
    script_dir = Path(__file__).parent.resolve()
    generate_script = script_dir / "generate_synthea_data.sh"
    
    # Make sure the script is executable
    if os.path.exists(generate_script):
        try:
            # Make the script executable
            os.chmod(generate_script, 0o755)
            
            # Execute the script
            import subprocess
            result = subprocess.run([generate_script], check=True, text=True, capture_output=True)
            
            logger.info(f"Synthea data generation complete: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating Synthea data: {e}")
            logger.error(f"Error output: {e.stderr}")
            logger.warning("You may need to generate Synthea data manually. See README for instructions.")
        except Exception as e:
            logger.error(f"Unexpected error generating Synthea data: {e}")
            logger.warning("You may need to generate Synthea data manually. See README for instructions.")
    else:
        logger.error(f"Synthea data generation script not found at: {generate_script}")
        logger.warning("You may need to generate Synthea data manually. See README for instructions.")

def main():
    """Main function to download all datasets."""
    logger.info("Starting dataset downloads")
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    download_pubmedqa()
    download_medmcqa()
    download_synthea()
    
    logger.info("All downloads complete")

if __name__ == "__main__":
    main()
