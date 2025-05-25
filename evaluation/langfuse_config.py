#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration for Langfuse evaluation logging.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langfuse configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Default project name
PROJECT_NAME = "MyOwn-MedAssist"

def validate_langfuse_config():
    """Validate Langfuse configuration."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("Warning: Langfuse API keys not found in environment variables.")
        print("To enable Langfuse logging, set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.")
        return False
    return True

def get_langfuse_config():
    """Get Langfuse configuration as a dictionary."""
    return {
        "public_key": LANGFUSE_PUBLIC_KEY,
        "secret_key": LANGFUSE_SECRET_KEY,
        "host": LANGFUSE_HOST,
        "project_name": PROJECT_NAME
    }

if __name__ == "__main__":
    # Simple test to validate configuration
    if validate_langfuse_config():
        print("Langfuse configuration is valid.")
        print(f"Host: {LANGFUSE_HOST}")
        print(f"Project: {PROJECT_NAME}")
    else:
        print("Langfuse configuration is incomplete.")
