#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit frontend for MyOwn-MedAssist:
1. Provides a user interface for medical question answering
2. Connects to the FastAPI backend
3. Supports token streaming
4. Offers an enhanced medical UI experience
"""

import os
import json
import requests
import streamlit as st
from typing import List, Dict, Any, Optional
import time
import random
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="MyOwn-MedAssist",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        color: #0d6efd;
        margin-bottom: 0.5rem;
        font-weight: 600;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Message styling */
    .user-message {
        background-color: #f0f2f5;
        padding: 1rem;
        border-radius: 15px 15px 15px 0px;
        margin: 1rem 2rem 1rem 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
    }
    
    .assistant-message {
        background-color: #e7f3fe;
        padding: 1rem;
        border-radius: 15px 15px 0px 15px;
        margin: 1rem 0.5rem 1rem 2rem;
        border-left: 4px solid #0d6efd;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        position: relative;
    }
    
    /* Source styling */
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        border: 1px solid #dee2e6;
        font-size: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .model-info {
        font-size: 0.8rem;
        color: #6c757d;
        text-align: right;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
        font-style: italic;
    }
    
    /* Example card styling */
    .example-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-color: #0d6efd;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    /* Chat input styling */
    .stTextInput>div>div>input {
        font-size: 1.1rem;
        border-radius: 20px;
        padding: 10px 15px;
    }
    
    /* Medical status indicators */
    .status-indicator {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    
    .status-circle {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.75em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 0.5rem;
    }
    
    .badge-primary {
        color: #fff;
        background-color: #0d6efd;
    }
    
    .badge-secondary {
        color: #fff;
        background-color: #6c757d;
    }
    
    .badge-info {
        color: #000;
        background-color: #0dcaf0;
    }
    
    /* Chat container */
    .chat-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Example medical questions for the interface
MEDICAL_EXAMPLES = [
    "What are the common symptoms of diabetes and how is it diagnosed?",
    "Can you explain the difference between Type 1 and Type 2 diabetes?",
    "What medications are typically prescribed for hypertension?",
    "What are the warning signs of a heart attack?",
    "How should I manage a fever in a 3-year-old child?",
    "What dietary changes can help lower cholesterol?",
    "What are the side effects of statins?",
    "How is rheumatoid arthritis different from osteoarthritis?",
    "What are the recommended vaccinations for a 65-year-old adult?",
    "What are the treatment options for chronic migraines?"
]

# Medical categories for organization
MEDICAL_CATEGORIES = {
    "Cardiology": ["heart", "cardiac", "cardiovascular", "blood pressure", "hypertension", "chest pain"],
    "Endocrinology": ["diabetes", "thyroid", "hormone", "insulin", "glucose"],
    "Neurology": ["brain", "headache", "migraine", "seizure", "stroke", "alzheimer"],
    "Gastroenterology": ["stomach", "intestine", "liver", "pancreas", "gallbladder", "digestive"],
    "Respiratory": ["lung", "asthma", "copd", "pneumonia", "bronchitis", "breathing"],
    "Pediatrics": ["child", "infant", "baby", "pediatric", "vaccination"],
    "Medications": ["drug", "medication", "pill", "prescription", "side effect", "dosage"]
}

def initialize_session_state():
    """Initialize session state variables."""
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize chat history for specific medical topics
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    
    # Track conversation topics
    if "conversation_topics" not in st.session_state:
        st.session_state.conversation_topics = []
    
    # Initialize model settings
    if "available_models" not in st.session_state:
        # Fetch available models from API
        try:
            response = requests.get(f"{API_URL}/models")
            if response.status_code == 200:
                data = response.json()
                st.session_state.available_models = data.get("models", ["gpt-3.5-turbo"])
                st.session_state.default_model = data.get("default", "gpt-3.5-turbo")
            else:
                st.session_state.available_models = ["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0125:personal::123abc"]
                st.session_state.default_model = "ft:gpt-3.5-turbo-0125:personal::123abc"  # Default to fine-tuned model
        except Exception as e:
            st.warning(f"Could not connect to API, using default models: {e}")
            st.session_state.available_models = ["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0125:personal::123abc"]
            st.session_state.default_model = "ft:gpt-3.5-turbo-0125:personal::123abc"  # Default to fine-tuned model
    
    # Initialize selected model
    if "selected_model" not in st.session_state:
        # Try to find a fine-tuned model and use it as default
        fine_tuned_models = [model for model in st.session_state.available_models if "ft:" in model]
        if fine_tuned_models:
            st.session_state.selected_model = fine_tuned_models[0]
        else:
            st.session_state.selected_model = st.session_state.default_model
    
    # Initialize RAG setting
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True  # Enable RAG by default for better medical responses
    
    # Initialize temperature setting
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.3  # Lower temperature for more precise medical information
    
    # Session-specific settings
    if "show_references" not in st.session_state:
        st.session_state.show_references = True
        
    if "auto_categorize" not in st.session_state:
        st.session_state.auto_categorize = True
        
    if "session_started" not in st.session_state:
        st.session_state.session_started = datetime.now().strftime("%Y-%m-%d %H:%M")
        
    if "queries_count" not in st.session_state:
        st.session_state.queries_count = 0

def get_medical_category(text):
    """Automatically categorize a medical question into a specialty."""
    text = text.lower()
    for category, keywords in MEDICAL_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return "General Medicine"

def display_header():
    """Display the application header with medical theming."""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="main-header">🏥 MyOwn-MedAssist</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Advanced Medical Question Answering System</div>', unsafe_allow_html=True)
    
    # Display session information
    with st.expander("Session Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Session Started:** {st.session_state.session_started}")
            st.markdown(f"**Queries Asked:** {st.session_state.queries_count}")
        with col2:
            if st.session_state.conversation_topics:
                st.markdown("**Topics Discussed:**")
                topics_html = ""
                for topic in set(st.session_state.conversation_topics):
                    topics_html += f'<span class="badge badge-primary">{topic}</span>'
                st.markdown(topics_html, unsafe_allow_html=True)
    
    # Medical disclaimer banner
    st.markdown(
        '<div class="disclaimer-box">⚠️ <strong>Medical Disclaimer:</strong> This system provides informational content only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</div>',
        unsafe_allow_html=True
    )

def display_medical_examples():
    """Display example medical questions that users can click on."""
    st.markdown("### Example Medical Questions")
    st.markdown("Click on any example to ask that question:")
    
    # Organize examples into columns
    col1, col2 = st.columns(2)
    half = len(MEDICAL_EXAMPLES) // 2
    
    # Create clickable example cards
    for i, example in enumerate(MEDICAL_EXAMPLES):
        # Get medical category for the example
        category = get_medical_category(example)
        
        # Create HTML for the example card
        example_html = f"""
        <div class="example-card" onclick="
        parent.postMessage({{event: 'streamlitClick', data: 'example_{i}'}}, '*')">
            <strong>{example}</strong>
            <br><span class="badge badge-secondary">{category}</span>
        </div>
        """
        
        # Display in alternating columns
        if i < half:
            col1.markdown(example_html, unsafe_allow_html=True)
        else:
            col2.markdown(example_html, unsafe_allow_html=True)

def display_sidebar():
    """Display enhanced sidebar with medical-specific settings."""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">MedAssist Settings</div>', unsafe_allow_html=True)
        
        # Create tabs for different settings categories
        tabs = st.tabs(["Model", "Features", "About"])
        
        # Model settings tab
        with tabs[0]:
            # Model selection with explanations
            st.markdown("#### Select Model")
            model_options = st.session_state.available_models
            model_descriptions = {
                "gpt-3.5-turbo": "Base model with general medical knowledge",
                "ft:gpt-3.5-turbo": "Fine-tuned on medical Q&A data"
            }
            
            # Create radio buttons with descriptions
            selected_index = 0
            for i, model in enumerate(model_options):
                if model == st.session_state.selected_model:
                    selected_index = i
                    break
                    
            st.session_state.selected_model = st.radio(
                "Available Models",
                options=model_options,
                index=selected_index,
                format_func=lambda x: f"{x} - {model_descriptions.get(x.split(':')[0], 'Specialized medical model')}"
            )
            
            # RAG toggle with explanation
            st.markdown("#### Knowledge Enhancement")
            rag_help = "RAG augments responses with information from medical literature"
            st.session_state.use_rag = st.toggle(
                "Enable Retrieval Augmented Generation", 
                value=st.session_state.use_rag,
                help=rag_help
            )
            
            # Show references toggle
            st.session_state.show_references = st.toggle(
                "Show Medical References", 
                value=st.session_state.show_references,
                help="Display sources of medical information when available"
            )
            
            # Temperature slider with medical context
            st.markdown("#### Response Style")
            st.session_state.temperature = st.slider(
                "Medical Precision",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Lower values for more precise medical information, higher for more creative responses"
            )
        
        # Features tab
        with tabs[1]:
            st.markdown("#### Conversation Features")
            
            # Auto-categorization toggle
            st.session_state.auto_categorize = st.toggle(
                "Auto-categorize Questions", 
                value=st.session_state.auto_categorize,
                help="Automatically categorize questions by medical specialty"
            )
            
            # Clear conversation button
            if st.button("Start New Consultation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_topics = []
                st.session_state.queries_count = 0
                st.session_state.session_started = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.rerun()
        
        # About tab
        with tabs[2]:
            st.markdown("#### About MyOwn-MedAssist")
            st.markdown(
                "MyOwn-MedAssist is an advanced medical question answering system that leverages "
                "fine-tuned language models and Retrieval Augmented Generation (RAG) "
                "to provide accurate and helpful medical information."
            )
            
            st.markdown("#### Capabilities")
            capabilities = [
                "🔍 Access to medical literature and guidelines",
                "📊 Evidence-based responses with references",
                "🧠 Fine-tuned on thousands of medical Q&A pairs",
                "📝 Detailed explanations of complex medical concepts",
                "🔄 Continuous learning from medical interactions"
            ]
            for cap in capabilities:
                st.markdown(f"- {cap}")
            
            st.markdown("#### Limitations")
            st.markdown(
                "⚠️ **Important Disclaimer**: This system is designed for informational purposes only. "
                "It is not a substitute for professional medical advice, diagnosis, or treatment. "
                "Always seek the advice of your physician or other qualified health provider "
                "with any questions you may have regarding a medical condition."
            )

def display_messages():
    """Display conversation messages with enhanced medical styling."""
    # Create a container for the chat history
    chat_container = st.container()
    
    with chat_container:
        st.markdown("### Medical Consultation")
        if not st.session_state.messages:
            # Display welcome message if no messages yet
            st.info("👋 Welcome to MyOwn-MedAssist! I'm here to answer your medical questions. How can I help you today?")
            return
            
        # Display each message in the conversation
        for i, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                # Get medical category for the user's question
                if st.session_state.auto_categorize:
                    category = get_medical_category(content)
                    category_badge = f'<span class="badge badge-info">{category}</span>'
                    st.markdown(f'<div class="user-message">👤 <strong>You</strong>: {category_badge} {content}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="user-message">👤 <strong>You</strong>: {content}</div>', unsafe_allow_html=True)
            else:
                # Display model info and assistant message
                model_info = message.get("model", st.session_state.selected_model)
                model_display = model_info.split(":")[0] if ":" in model_info else model_info
                st.markdown(f'<div class="model-info">Model: {model_display}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="assistant-message">🏥 <strong>MedAssist</strong>: {content}</div>', unsafe_allow_html=True)
                
                # Display sources if available and enabled
                if "sources" in message and message["sources"] and st.session_state.show_references:
                    with st.expander("📚 Medical References"):
                        for i, source in enumerate(message["sources"]):
                            source_text = source.get("text", "")
                            source_name = source.get("source", "Unknown")
                            source_url = source.get("url", "#")
                            
                            if len(source_text) > 300:
                                source_text = source_text[:300] + "..."
                            
                            st.markdown(f'<div class="source-box">{source_text}<br><small><em>Source: <a href="{source_url}" target="_blank">{source_name}</a></em></small></div>', unsafe_allow_html=True)

def handle_example_click():
    """Handle click events from example medical questions."""
    # Check query parameters for example clicks
    clicked_example = st.query_params.get("example", None)
    
    if clicked_example and clicked_example.startswith("example_"):
        try:
            example_index = int(clicked_example.split("_")[1])
            if 0 <= example_index < len(MEDICAL_EXAMPLES):
                # Send the example question as a message
                send_message(MEDICAL_EXAMPLES[example_index])
                # Clear the query parameter to avoid repeated processing
                st.query_params.clear()
        except ValueError:
            pass

def mock_api_response(user_input: str):
    """Generate a mock API response when the real API is not available."""
    # This is a fallback function to simulate API responses
    # It allows testing the UI even if the backend isn't running
    
    # Determine medical category
    category = get_medical_category(user_input)
    
    # Add to conversation topics
    if category not in st.session_state.conversation_topics:
        st.session_state.conversation_topics.append(category)
    
    # Simple keyword-based responses for common medical questions
    responses = {
        "diabetes": "Diabetes is a chronic condition characterized by high blood sugar levels. Common symptoms include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision. Diagnosis typically involves blood tests measuring fasting blood glucose, HbA1c levels, and sometimes an oral glucose tolerance test.",
        "hypertension": "Hypertension (high blood pressure) is often treated with several classes of medications including diuretics, ACE inhibitors, ARBs, calcium channel blockers, and beta-blockers. Lifestyle modifications like reducing sodium intake, regular exercise, maintaining healthy weight, limiting alcohol, and stress management are also essential components of treatment.",
        "heart attack": "Warning signs of a heart attack include chest pain or discomfort (feeling of pressure, squeezing, fullness or pain), pain radiating to the jaw, neck, back, arm or shoulder, shortness of breath, cold sweat, nausea, lightheadedness, and unusual fatigue. Women may experience less typical symptoms like shortness of breath, nausea/vomiting, and back or jaw pain. If you suspect a heart attack, call emergency services immediately as prompt treatment is crucial.",
        "cholesterol": "Dietary changes to lower cholesterol include reducing saturated and trans fats, increasing soluble fiber (found in oats, beans, fruits), adding plant sterols and stanols, consuming fatty fish rich in omega-3 fatty acids, limiting dietary cholesterol, and maintaining a healthy weight through regular physical activity."
    }
    
    # Check for keywords in the question
    response_text = "I don't have specific information on that medical topic. As a medical assistant, I'd recommend consulting with a healthcare professional for personalized advice."
    
    for keyword, response in responses.items():
        if keyword in user_input.lower():
            response_text = response
            break
    
    # Add a disclaimer
    response_text += "\n\nPlease note that this information is for educational purposes only and not a substitute for professional medical advice."
    
    # Simulate response delay
    time.sleep(1)
    
    return {
        "content": response_text,
        "model": st.session_state.selected_model,
        "sources": []
    }

def send_message(user_input: str):
    """Send message to API and handle response."""
    if not user_input.strip():
        return
    
    # Update query count
    st.session_state.queries_count += 1
    
    # Add user message to conversation
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Get medical category and update conversation topics
    category = get_medical_category(user_input)
    if category not in st.session_state.conversation_topics:
        st.session_state.conversation_topics.append(category)
    
    # Prepare messages for API
    api_messages = []
    for msg in st.session_state.messages:
        if msg.get("sources") is not None:
            # Skip metadata that's not meant for the API
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        else:
            api_messages.append(msg)
    
    # Create placeholder for streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        model_info = st.session_state.selected_model
        sources = []
        
        try:
            # Check if API URL is accessible
            api_available = True
            try:
                # Try a quick connection to check API availability
                response = requests.get(f"{API_URL}/models", timeout=2)
                api_available = response.status_code == 200
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                api_available = False
            
            if not api_available:
                # Use mock response if API is not available
                st.warning("API not available. Using simulated responses for demonstration.")
                # Simulate streaming response
                mock_response = mock_api_response(user_input)
                
                # Display typing indicator
                for i in range(min(len(mock_response["content"]) // 3, 10)):
                    dots = "." * (i % 4)
                    message_placeholder.markdown(f"*Thinking{dots}*")
                    time.sleep(0.3)
                
                # Stream the mock response character by character
                for char in mock_response["content"]:
                    full_response += char
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
                
                # Set model info and sources from mock response
                model_info = mock_response["model"]
                sources = mock_response["sources"]
            else:
                # Make streaming API request to the real backend
                with requests.post(
                    f"{API_URL}/chat/stream",
                    json={
                        "messages": api_messages,
                        "model": st.session_state.selected_model,
                        "use_rag": st.session_state.use_rag,
                        "stream": True,
                        "temperature": st.session_state.temperature
                    },
                    stream=True
                ) as response:
                    # Check for errors
                    if response.status_code != 200:
                        error_msg = f"Error: API returned status code {response.status_code}"
                        try:
                            error_data = response.json()
                            if "detail" in error_data:
                                error_msg += f" - {error_data['detail']}"
                        except:
                            pass
                        message_placeholder.error(error_msg)
                        return
                    
                    # Process streaming response
                    for line in response.iter_lines():
                        if line:
                            try:
                                line_data = json.loads(line)
                                
                                # Handle model info
                                if "model" in line_data:
                                    model_info = line_data["model"]
                                
                                # Handle token
                                if "token" in line_data:
                                    token = line_data["token"]
                                    full_response += token
                                    message_placeholder.markdown(full_response + "▌")
                                
                                # Handle sources
                                if "sources" in line_data:
                                    sources = line_data["sources"]
                                
                                # Handle errors
                                if "error" in line_data:
                                    message_placeholder.error(f"Error: {line_data['error']}")
                                    return
                                
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")
            return
        
        # Update with final response
        message_placeholder.markdown(full_response)
    
    # Add assistant message to conversation
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "model": model_info,
        "sources": sources
    })
    
    # Force a rerun to update the UI
    st.rerun()

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Handle example clicks from URL parameters
    handle_example_click()
    
    # Display header with medical branding
    display_header()
    
    # Display sidebar with settings
    display_sidebar()
    
    # Main content area - two columns layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat container to display messages
        with st.container(height=500, border=False):
            display_messages()
        
        # User input at the bottom
        user_input = st.chat_input("Ask a medical question...")
        if user_input:
            send_message(user_input)
    
    with col2:
        # Display example medical questions
        display_medical_examples()
        
        # Medical topics reference
        with st.expander("Medical Topics Reference", expanded=False):
            st.markdown("### Medical Specialties")
            for specialty, keywords in MEDICAL_CATEGORIES.items():
                st.markdown(f"**{specialty}**: {', '.join(keywords[:3])}...")
        
        # Medical resources
        with st.expander("Additional Medical Resources", expanded=False):
            st.markdown("### Trusted Resources")
            resources = [
                ("CDC", "https://www.cdc.gov/", "Disease information and prevention guidelines"),
                ("Mayo Clinic", "https://www.mayoclinic.org/", "Comprehensive medical information"),
                ("MedlinePlus", "https://medlineplus.gov/", "Health information from the National Library of Medicine"),
                ("WHO", "https://www.who.int/", "Global health guidelines and information")
            ]
            
            for name, url, desc in resources:
                st.markdown(f"* **[{name}]({url})** - {desc}")
            
            st.markdown("\n⚠️ *Always consult healthcare professionals for medical advice.*")

# Add event handlers for the interface components
def handle_streamlit_events():
    """Handle various Streamlit UI events."""
    # This is needed for the clickable example cards
    st.markdown("""
    <script>
    const streamlitDoc = window.parent.document;
    
    const buttons = streamlitDoc.querySelectorAll('.example-card');
    buttons.forEach((button, index) => {{
        button.addEventListener('click', () => {{
            // Use the new URL approach for query parameters
            const currentUrl = new URL(window.parent.location);
            currentUrl.searchParams.set('example', `example_${{index}}`);
            window.parent.history.pushState({}, '', currentUrl);
            window.parent.location.reload();
        }});
    }});
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
