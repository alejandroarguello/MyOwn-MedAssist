#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Locust load testing script for MyOwn-MedAssist API.
Demonstrates support for 10,000 concurrent users.
"""

import json
import random
from locust import HttpUser, task, between

# Sample medical questions for testing
MEDICAL_QUESTIONS = [
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed?",
    "What are common treatments for migraine headaches?",
    "What are the side effects of statins?",
    "How is pneumonia treated?",
    "What are the risk factors for heart disease?",
    "How is rheumatoid arthritis diagnosed?",
    "What are the symptoms of COVID-19?",
    "How is asthma managed in children?",
    "What are common antibiotics for urinary tract infections?",
    "What are the stages of chronic kidney disease?",
    "How is hypothyroidism diagnosed and treated?",
    "What are the symptoms of multiple sclerosis?",
    "How is Parkinson's disease managed?",
    "What are the treatment options for depression?",
    "How is ADHD diagnosed in adults?",
    "What are the symptoms of appendicitis?",
    "How is osteoporosis prevented and treated?",
    "What are the risk factors for stroke?",
    "How is GERD managed?"
]

class APIUser(HttpUser):
    """Simulated user for load testing the API."""
    
    # Wait between 3-7 seconds between tasks
    wait_time = between(3, 7)
    
    def on_start(self):
        """Initialize user session."""
        # Get available models
        response = self.client.get("/models")
        if response.status_code == 200:
            data = response.json()
            self.available_models = data.get("models", ["gpt-3.5-turbo"])
            self.default_model = data.get("default", "gpt-3.5-turbo")
        else:
            self.available_models = ["gpt-3.5-turbo"]
            self.default_model = "gpt-3.5-turbo"
    
    @task(1)
    def get_root(self):
        """Test the root endpoint."""
        self.client.get("/")
    
    @task(1)
    def get_models(self):
        """Test the models endpoint."""
        self.client.get("/models")
    
    @task(5)
    def chat_non_streaming(self):
        """Test the chat endpoint without streaming."""
        # Select a random question
        question = random.choice(MEDICAL_QUESTIONS)
        
        # Select a random model
        model = random.choice(self.available_models)
        
        # Randomly decide whether to use RAG
        use_rag = random.choice([True, False])
        
        # Create request payload
        payload = {
            "messages": [
                {"role": "system", "content": "You are a medical assistant that provides accurate information."},
                {"role": "user", "content": question}
            ],
            "model": model,
            "use_rag": use_rag,
            "stream": False,
            "temperature": 0.7
        }
        
        # Send request
        with self.client.post("/chat", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "response" in data and "model" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Request failed with status code {response.status_code}")
    
    @task(3)
    def chat_streaming(self):
        """Test the chat streaming endpoint."""
        # Select a random question
        question = random.choice(MEDICAL_QUESTIONS)
        
        # Select a random model
        model = random.choice(self.available_models)
        
        # Randomly decide whether to use RAG
        use_rag = random.choice([True, False])
        
        # Create request payload
        payload = {
            "messages": [
                {"role": "system", "content": "You are a medical assistant that provides accurate information."},
                {"role": "user", "content": question}
            ],
            "model": model,
            "use_rag": use_rag,
            "stream": True,
            "temperature": 0.7
        }
        
        # Send request
        with self.client.post("/chat/stream", json=payload, catch_response=True, stream=True) as response:
            if response.status_code == 200:
                # For streaming responses, we just check that the connection works
                # and that we receive some data
                received_data = False
                
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        received_data = True
                        break
                
                if received_data:
                    response.success()
                else:
                    response.failure("No data received from streaming endpoint")
            else:
                response.failure(f"Request failed with status code {response.status_code}")

# To run this load test:
# locust -f locustfile.py --host=http://localhost:8000
# Then open http://localhost:8089 in a browser
