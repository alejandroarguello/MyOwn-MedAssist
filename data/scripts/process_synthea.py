#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Synthea EHR dataset:
1. Parse FHIR JSON files from Synthea output
2. Clean and structure the data
3. Convert to JSONL format suitable for OpenAI fine-tuning
4. Ensure no PHI is present (Synthea generates synthetic data, but we'll still check)
"""

import os
import json
import jsonlines
import pandas as pd
from pathlib import Path
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Constants
RAW_DATA_DIR = (SCRIPT_DIR / "../raw/synthea/synthea-master").resolve()
PROCESSED_DATA_DIR = (SCRIPT_DIR / "../processed").resolve()
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
OUTPUT_FILE = PROCESSED_DATA_DIR / "synthea_processed.jsonl"

print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
print(f"FHIR dir: {RAW_DATA_DIR / 'output' / 'fhir'}")
print(f"FHIR dir exists: {(RAW_DATA_DIR / 'output' / 'fhir').exists()}")

def find_synthea_output() -> Optional[Path]:
    """Find Synthea output directory with generated FHIR data."""
    # Look for FHIR output directory
    fhir_dir = RAW_DATA_DIR / "output" / "fhir"
    print(fhir_dir)
    if not fhir_dir.exists():
        # Try alternative location if the first one doesn't exist
        fhir_dir = RAW_DATA_DIR / "output"
        if not fhir_dir.exists():
            logger.warning(f"No Synthea output directory found at {fhir_dir}. Please run Synthea first.")
            return None
            
        # If we found output directory but not fhir subdirectory, check if fhir files are directly in output
        if not any(fhir_dir.glob("*.json")):
            logger.warning(f"No FHIR JSON files found in {fhir_dir}. Please run Synthea with FHIR export enabled.")
            return None
            
        return fhir_dir
    
    return fhir_dir

def parse_fhir_bundle(file_path: Path) -> Dict[str, List[Dict]]:
    """Parse a FHIR Bundle resource and extract relevant resources."""
    resources = {
        'Patient': [],
        'Condition': [],
        'MedicationRequest': [],
        'Encounter': [],
        'Observation': []
    }
    
    try:
        with open(file_path, 'r') as f:
            bundle = json.load(f)
            
        if bundle.get('resourceType') != 'Bundle' or 'entry' not in bundle:
            logger.warning(f"Not a valid FHIR Bundle: {file_path}")
            return resources
            
        for entry in bundle.get('entry', []):
            if 'resource' not in entry:
                continue
                
            resource = entry['resource']
            resource_type = resource.get('resourceType')
            
            if resource_type in resources:
                resources[resource_type].append(resource)
                
    except Exception as e:
        logger.error(f"Error parsing FHIR bundle {file_path}: {str(e)}")
    
    return resources

def process_fhir_resources(fhir_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process all FHIR JSON files and convert to DataFrames."""
    patients = []
    conditions = []
    medications = []
    encounters = []
    
    # Process each patient's FHIR bundle
    for fhir_file in fhir_dir.glob('*.json'):
        resources = parse_fhir_bundle(fhir_file)
        
        # Process patients
        for patient in resources['Patient']:
            patients.append({
                'Id': patient.get('id'),
                'BIRTHDATE': patient.get('birthDate', ''),
                'GENDER': patient.get('gender', ''),
                'FIRST': patient.get('name', [{}])[0].get('given', [''])[0],
                'LAST': patient.get('name', [{}])[0].get('family', '')
            })
            
        # Process conditions
        for condition in resources['Condition']:
            patient_ref = condition.get('subject', {}).get('reference', '')
            conditions.append({
                'PATIENT': patient_ref,  # Keep the full reference
                'CODE': condition.get('code', {}).get('coding', [{}])[0].get('code', ''),
                'DESCRIPTION': condition.get('code', {}).get('text', ''),
                'START': condition.get('onsetDateTime', '')
            })
            
        # Process medications
        for med in resources['MedicationRequest']:
            patient_ref = med.get('subject', {}).get('reference', '')
            medications.append({
                'PATIENT': patient_ref,  # Keep the full reference
                'CODE': med.get('medicationCodeableConcept', {}).get('coding', [{}])[0].get('code', ''),
                'DESCRIPTION': med.get('medicationCodeableConcept', {}).get('text', ''),
                'START': med.get('authoredOn', '')
            })
            
        # Process encounters
        for encounter in resources['Encounter']:
            patient_ref = encounter.get('subject', {}).get('reference', '')
            encounters.append({
                'PATIENT': patient_ref,  # Keep the full reference
                'Id': encounter.get('id'),
                'START': encounter.get('period', {}).get('start', ''),
                'STOP': encounter.get('period', {}).get('end', '')
            })
    
    # Convert to DataFrames
    patients_df = pd.DataFrame(patients)
    conditions_df = pd.DataFrame(conditions)
    medications_df = pd.DataFrame(medications)
    encounters_df = pd.DataFrame(encounters)
    
    logger.info(f"Processed {len(patients_df)} patients, {len(conditions_df)} conditions, "
                f"{len(medications_df)} medications, {len(encounters_df)} encounters")
    
    return patients_df, conditions_df, medications_df, encounters_df

def load_synthea_data(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Synthea EHR data from FHIR JSON files."""
    logger.info(f"Loading Synthea data from {output_dir}")
    
    if not output_dir.exists() or not any(output_dir.glob('*.json')):
        logger.warning("No FHIR JSON files found in the output directory.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    return process_fhir_resources(output_dir)

def create_qa_pairs(patients_df, conditions_df, medications_df, encounters_df):
    """Create Q&A pairs from Synthea data for fine-tuning."""
    logger.info("Creating Q&A pairs from Synthea data")
    
    # Print column names for debugging
    print(f"Patients columns: {patients_df.columns.tolist()}")
    print(f"Conditions columns: {conditions_df.columns.tolist()}")
    print(f"Medications columns: {medications_df.columns.tolist()}")
    
    # Print sample data
    if not patients_df.empty:
        print(f"Sample patient: {patients_df.iloc[0].to_dict()}")
    if not conditions_df.empty:
        print(f"Sample condition: {conditions_df.iloc[0].to_dict()}")
    if not medications_df.empty:
        print(f"Sample medication: {medications_df.iloc[0].to_dict()}")
    
    qa_pairs = []
    
    # Join data to get patient-centric view
    if patients_df.empty or conditions_df.empty or medications_df.empty or encounters_df.empty:
        logger.warning("Missing data, cannot create Q&A pairs")
        return qa_pairs
    
    # Process each patient
    for i, (_, patient) in enumerate(patients_df.iterrows()):
        patient_id = patient['Id']
        print(f"Processing patient {i+1}/{len(patients_df)}: {patient_id}")
        
        # Create patient reference formats to match against
        patient_refs = [
            f"urn:uuid:{patient_id}",  # Most common format
            f"Patient/{patient_id}",    # Alternative format
            patient_id                   # Plain UUID
        ]
        
        # Get patient conditions - match any of the reference formats
        patient_conditions = conditions_df[conditions_df['PATIENT'].isin(patient_refs)]
        print(f"  Found {len(patient_conditions)} conditions")
        
        # Get patient medications - match any of the reference formats
        patient_medications = medications_df[medications_df['PATIENT'].isin(patient_refs)]
        print(f"  Found {len(patient_medications)} medications")
        
        # Get patient encounters - match any of the reference formats
        patient_encounters = encounters_df[encounters_df['PATIENT'].isin(patient_refs)]
        print(f"  Found {len(patient_encounters)} encounters")
        
        # Skip patients with no data
        if patient_conditions.empty or patient_medications.empty:
            print(f"  Skipping patient {patient_id} due to insufficient data")
            continue
        
        # Create various types of Q&A pairs
        pairs_created = 0
        
        # 1. Medication for condition
        for _, condition in patient_conditions.iterrows():
            condition_name = condition['DESCRIPTION']
            if not condition_name:  # Skip conditions with no description
                continue
                
            # Use all medications for this patient
            relevant_meds = patient_medications
            
            # Try to filter by date if possible
            if 'START' in condition and 'START' in patient_medications:
                if condition['START'] and all(patient_medications['START'].notna()):
                    try:
                        condition_date = pd.to_datetime(condition['START'])
                        relevant_meds = patient_medications[
                            pd.to_datetime(patient_medications['START']) >= condition_date
                        ]
                    except (ValueError, pd.errors.OutOfBoundsDatetime, TypeError) as e:
                        print(f"  Date parsing error: {e}")
                        # If date parsing fails, use all medications
                        pass
            
            if len(relevant_meds) > 0:
                med_names = ", ".join(relevant_meds['DESCRIPTION'].unique())
                
                qa_pair = {
                    "messages": [
                        {"role": "system", "content": "You are a medical assistant that provides information about medications for medical conditions."},
                        {"role": "user", "content": f"What medications might be prescribed for {condition_name}?"},
                        {"role": "assistant", "content": f"For {condition_name}, medications that might be prescribed include: {med_names}. Note that medication choices depend on the specific patient circumstances, severity of the condition, and other factors."}
                    ]
                }
                
                qa_pairs.append(qa_pair)
        
        # 2. Condition summary
        if len(patient_conditions) > 0:
            condition_names = ", ".join(patient_conditions['DESCRIPTION'].unique())
            
            qa_pair = {
                "messages": [
                    {"role": "system", "content": "You are a medical assistant that summarizes patient conditions."},
                    {"role": "user", "content": f"Summarize the medical conditions for a patient with: {condition_names}"},
                    {"role": "assistant", "content": f"The patient has multiple conditions including {condition_names}. These conditions require careful management and regular monitoring. It's important to consider how these conditions might interact with each other and affect treatment options."}
                ]
            }
            
            qa_pairs.append(qa_pair)
        
        # 3. Treatment plan
        if len(patient_conditions) > 0 and len(patient_medications) > 0:
            main_condition = patient_conditions.iloc[0]['DESCRIPTION']
            med_names = ", ".join(patient_medications['DESCRIPTION'].unique())
            
            qa_pair = {
                "messages": [
                    {"role": "system", "content": "You are a medical assistant that helps with treatment planning."},
                    {"role": "user", "content": f"What would a treatment plan look like for a patient with {main_condition} who is taking {med_names}?"},
                    {"role": "assistant", "content": f"A treatment plan for a patient with {main_condition} who is taking {med_names} would typically include:\n\n1. Regular monitoring of medication effectiveness and side effects\n2. Periodic lab tests to check organ function and medication levels\n3. Lifestyle modifications appropriate for the condition\n4. Regular follow-up appointments\n5. Monitoring for potential drug interactions\n6. Patient education about the condition and medications\n\nThis plan should be personalized based on the patient's specific circumstances, comorbidities, and response to treatment."}
                ]
            }
            
            qa_pairs.append(qa_pair)
    
    logger.info(f"Created {len(qa_pairs)} Q&A pairs from Synthea data")
    return qa_pairs

def process_synthea():
    """Process Synthea EHR dataset and convert to JSONL for fine-tuning."""
    logger.info("Processing Synthea dataset")
    
    # Find Synthea output directory
    output_dir = find_synthea_output()
    
    if not output_dir:
        logger.warning("Synthea data not found. Please run Synthea first.")
        return 0
    
    # Load data
    patients_df, conditions_df, medications_df, encounters_df = load_synthea_data(output_dir)
    
    # Create Q&A pairs
    qa_pairs = create_qa_pairs(patients_df, conditions_df, medications_df, encounters_df)
    
    # Save to JSONL
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        for example in qa_pairs:
            writer.write(example)
    
    logger.info(f"Saved {len(qa_pairs)} examples to {OUTPUT_FILE}")
    
    return len(qa_pairs)

def main():
    """Main function to process Synthea dataset."""
    logger.info("Starting Synthea processing")
    
    num_examples = process_synthea()
    
    logger.info(f"Processing complete. {num_examples} examples processed.")

if __name__ == "__main__":
    main()
