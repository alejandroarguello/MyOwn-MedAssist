#!/bin/bash

# Script to generate Synthea synthetic EHR data
# This should be run after running download_datasets.py

set -e  # Exit on error

# Navigate to the project root
cd "$(dirname "$0")/../.."

# Define paths
SYNTHEA_DIR="data/raw/synthea"
SYNTHEA_JAR="synthea-master/build/libs/synthea-with-dependencies.jar"
OUTPUT_DIR="data/raw/synthea/output"

# Check if Java is installed
if ! command -v java &> /dev/null; then
    echo "Java is required but not installed. Please install OpenJDK 11 or later."
    echo "On macOS: brew install openjdk@11"
    echo "On Ubuntu/Debian: sudo apt-get install openjdk-11-jdk"
    exit 1
fi

# Check if Synthea was downloaded
if [ ! -d "$SYNTHEA_DIR/synthea-master" ]; then
    echo "Synthea not found. Please run download_datasets.py first."
    exit 1
fi

# Navigate to Synthea directory
cd "$SYNTHEA_DIR/synthea-master"

# Build Synthea if not already built
if [ ! -f "$SYNTHEA_JAR" ]; then
    echo "Building Synthea..."
    ./gradlew build check test
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate synthetic patient data
echo "Generating synthetic EHR data for 1000 patients..."
java -jar "$SYNTHEA_JAR" \
    -p 1000 \
    -o "$OUTPUT_DIR" \
    --exporter.fhir.export true \
    --exporter.fhir.transaction_bundle true \
    --exporter.csv.export true

echo "Synthetic data generation complete. Output saved to: $OUTPUT_DIR"

# Return to original directory
cd - > /dev/null
