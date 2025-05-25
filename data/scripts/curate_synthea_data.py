#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Curate Synthea EHR dataset:
1. Filter out non-medical conditions
2. Implement proper medication-condition mapping
3. Add clinical context
4. Improve data quality
"""

import os
import json
import jsonlines
from pathlib import Path
import logging
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
PROCESSED_DATA_DIR = (SCRIPT_DIR / "../processed").resolve()
INPUT_FILE = PROCESSED_DATA_DIR / "synthea_processed.jsonl"
OUTPUT_FILE = PROCESSED_DATA_DIR / "synthea_curated.jsonl"

# Medical condition whitelist and medication mapping
CONDITION_MED_MAP = {
    # Respiratory conditions
    "Acute bronchitis (disorder)": {
        "medications": ["Azithromycin", "Amoxicillin", "Doxycycline", "Ibuprofen", "Acetaminophen"],
        "context": "Antibiotics may be prescribed if bacterial infection is suspected. Pain relievers can help with fever and discomfort."
    },
    "Viral sinusitis (disorder)": {
        "medications": ["Acetaminophen", "Ibuprofen", "Pseudoephedrine", "Saline nasal spray"],
        "context": "Treatment focuses on symptom relief as antibiotics are not effective against viral infections."
    },
    "Acute bacterial sinusitis (disorder)": {
        "medications": ["Amoxicillin", "Amoxicillin-Clavulanate", "Doxycycline", "Levofloxacin"],
        "context": "Antibiotics are prescribed to treat the bacterial infection. Treatment typically lasts 7-14 days."
    },
    "Streptococcal sore throat (disorder)": {
        "medications": ["Penicillin V Potassium", "Amoxicillin", "Azithromycin", "Acetaminophen"],
        "context": "Antibiotics are essential to prevent complications like rheumatic fever. Complete the full course even if symptoms improve."
    },
    "Acute viral pharyngitis (disorder)": {
        "medications": ["Acetaminophen", "Ibuprofen", "Throat lozenges"],
        "context": "Treatment is supportive as antibiotics are not effective against viral infections."
    },
    
    # Cardiovascular conditions
    "Essential hypertension (disorder)": {
        "medications": ["Hydrochlorothiazide", "Lisinopril", "Amlodipine", "Metoprolol"],
        "context": "First-line treatments often include diuretics, ACE inhibitors, calcium channel blockers, or beta-blockers."
    },
    
    # Dental/oral conditions
    "Gingivitis (disorder)": {
        "medications": ["Chlorhexidine gluconate mouthwash", "Hydrogen peroxide rinse"],
        "context": "Treatment focuses on improving oral hygiene and reducing inflammation."
    },
    "Gingival disease (disorder)": {
        "medications": ["Chlorhexidine gluconate mouthwash", "Hydrogen peroxide rinse", "Antibiotics"],
        "context": "More severe than gingivitis, may require antibiotics if infection is present."
    },
    "Loss of teeth (disorder)": {
        "medications": ["Antibiotics (post-extraction)", "Acetaminophen", "Ibuprofen"],
        "context": "Pain management is key after tooth loss or extraction. Antibiotics may be prescribed to prevent infection."
    },
    
    # Pain conditions
    "Chronic pain (finding)": {
        "medications": ["Acetaminophen", "Ibuprofen", "Naproxen", "Duloxetine", "Gabapentin"],
        "context": "Treatment approach depends on the cause, severity, and duration of pain."
    },
    "Chronic low back pain (finding)": {
        "medications": ["Acetaminophen", "Ibuprofen", "Naproxen", "Cyclobenzaprine", "Duloxetine"],
        "context": "NSAIDs are first-line for pain relief. Muscle relaxants may be added for muscle spasms."
    },
    
    # Injuries
    "Injury of knee (disorder)": {
        "medications": ["Ibuprofen", "Naproxen", "Acetaminophen"],
        "context": "NSAIDs help reduce pain and inflammation. Physical therapy is often recommended."
    },
    "Injury of anterior cruciate ligament (disorder)": {
        "medications": ["Ibuprofen", "Naproxen", "Acetaminophen", "Opioids (short-term)"],
        "context": "Pain management is important initially. Surgical repair may be needed depending on severity."
    },
    
    # Other conditions
    "Anemia (disorder)": {
        "medications": ["Ferrous sulfate", "Vitamin B12", "Folic acid"],
        "context": "Treatment depends on the type of anemia. Iron supplements are common for iron deficiency anemia."
    },
    "Body mass index 30+ - obesity (finding)": {
        "medications": ["Orlistat", "Phentermine-topiramate", "Naltrexone-bupropion"],
        "context": "Medications are typically used alongside lifestyle modifications including diet and exercise."
    },
    "Dependent drug abuse (disorder)": {
        "medications": ["Buprenorphine", "Methadone", "Naltrexone"],
        "context": "Medication-assisted treatment can help manage withdrawal symptoms and cravings."
    },
    "Overdose (disorder)": {
        "medications": ["Naloxone (for opioid overdose)", "Flumazenil (for benzodiazepine overdose)"],
        "context": "Immediate medical attention is required. Specific antidotes depend on the substance involved."
    },
    "Normal pregnancy (finding)": {
        "medications": ["Prenatal vitamins", "Folic acid", "Iron supplements"],
        "context": "Nutritional supplements are recommended to support fetal development and maternal health."
    },
    "Stress (finding)": {
        "medications": ["SSRIs", "Benzodiazepines (short-term)"],
        "context": "Medication may be prescribed if stress leads to anxiety or depression. Therapy is often recommended."
    }
}

# Non-medical conditions to filter out
NON_MEDICAL_CONDITIONS = [
    "Housing unsatisfactory (finding)",
    "Received higher education (finding)",
    "Part-time employment (finding)",
    "Full-time employment (finding)",
    "Limited social contact (finding)",
    "Social isolation (finding)",
    "Reports of violence in the environment (finding)",
    "Has a criminal record (finding)",
    "Not in labor force (finding)",
    "Medication review due (situation)"  # This is a situation, not a condition requiring medication
]

def is_medical_condition(condition):
    """Check if the condition is a legitimate medical condition."""
    # Filter out explicitly non-medical conditions
    if condition in NON_MEDICAL_CONDITIONS:
        return False
    
    # Check if it's in our whitelist
    if condition in CONDITION_MED_MAP:
        return True
    
    # Additional heuristics for conditions not in our map
    non_medical_terms = ["employment", "education", "housing", "criminal", "social", "labor force"]
    if any(term in condition.lower() for term in non_medical_terms):
        return False
    
    # If it has "disorder", "disease", "injury", "pain", it's likely medical
    medical_terms = ["disorder", "disease", "injury", "pain", "syndrome", "infection"]
    if any(term in condition.lower() for term in medical_terms):
        return True
    
    # Default to including it if we're not sure
    return True

def get_appropriate_medications(condition):
    """Get medically appropriate medications for a condition."""
    if condition in CONDITION_MED_MAP:
        return CONDITION_MED_MAP[condition]["medications"]
    return []

def get_medication_context(condition):
    """Get clinical context for medications for a condition."""
    if condition in CONDITION_MED_MAP:
        return CONDITION_MED_MAP[condition]["context"]
    return "Medication choices depend on the specific patient circumstances, severity of the condition, and other factors."

def format_medication_response(condition, medications):
    """Format a response about medications for a condition."""
    if not medications:
        return f"For {condition}, specific medications would be determined by a healthcare provider based on individual assessment. This condition may not typically require medication, or may need specialized treatment."
    
    med_list = ", ".join(medications)
    context = get_medication_context(condition)
    
    return f"For {condition}, medications that might be prescribed include: {med_list}. {context}"

def curate_qa_pair(qa_pair):
    """Curate a single Q&A pair for better quality."""
    messages = qa_pair.get("messages", [])
    if len(messages) != 3:  # system, user, assistant
        return None
    
    system_msg, user_msg, assistant_msg = messages
    
    # Check if this is a medication Q&A
    if "What medications might be prescribed for" not in user_msg.get("content", ""):
        return qa_pair  # Keep non-medication Q&As as is
    
    # Extract the condition
    condition_match = re.search(r"What medications might be prescribed for (.+?)\?", user_msg.get("content", ""))
    if not condition_match:
        return None
    
    condition = condition_match.group(1)
    
    # Filter out non-medical conditions
    if not is_medical_condition(condition):
        logger.info(f"Filtering out non-medical condition: {condition}")
        return None
    
    # Get appropriate medications
    medications = get_appropriate_medications(condition)
    
    # Create curated response
    curated_response = format_medication_response(condition, medications)
    
    # Update the assistant message
    assistant_msg["content"] = curated_response
    
    return {"messages": [system_msg, user_msg, assistant_msg]}

def curate_synthea_data():
    """Curate the Synthea processed data for better quality."""
    logger.info(f"Curating Synthea data from {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        logger.error(f"Input file {INPUT_FILE} not found")
        return 0
    
    qa_pairs = []
    curated_count = 0
    filtered_count = 0
    condition_counts = {}
    
    # Read the input file
    with jsonlines.open(INPUT_FILE) as reader:
        for qa_pair in reader:
            # Extract condition for logging
            messages = qa_pair.get("messages", [])
            if len(messages) >= 2:
                user_msg = messages[1]
                condition_match = re.search(r"What medications might be prescribed for (.+?)\?", user_msg.get("content", ""))
                if condition_match:
                    condition = condition_match.group(1)
                    condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            # Curate the Q&A pair
            curated_pair = curate_qa_pair(qa_pair)
            if curated_pair:
                qa_pairs.append(curated_pair)
                curated_count += 1
            else:
                filtered_count += 1
    
    # Write the curated data
    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        for qa_pair in qa_pairs:
            writer.write(qa_pair)
    
    # Log condition counts
    logger.info("Condition counts:")
    for condition, count in sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        logger.info(f"  {condition}: {count}")
    
    logger.info(f"Curated {curated_count} Q&A pairs, filtered out {filtered_count} pairs")
    logger.info(f"Saved curated data to {OUTPUT_FILE}")
    
    return curated_count

def main():
    """Main function to curate Synthea data."""
    logger.info("Starting Synthea data curation")
    
    num_curated = curate_synthea_data()
    
    logger.info(f"Curation complete. {num_curated} Q&A pairs curated.")

if __name__ == "__main__":
    main()
