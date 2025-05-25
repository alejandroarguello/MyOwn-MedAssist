#!/bin/bash

# Setup script for MyOwn-MedAssist Anaconda environment
echo "Setting up Anaconda environment for MyOwn-MedAssist..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'medassist' with Python 3.10..."
conda create -y -n medassist python=3.10

# Activate environment
echo "Activating environment..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate medassist

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file to add your API keys."
fi

echo "Setup complete! To activate the environment, run:"
echo "conda activate medassist"
