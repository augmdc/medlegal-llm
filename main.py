import streamlit as st
import PyPDF2
import io
import requests
import json
import subprocess
import atexit
import signal
import time
import os
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Global variable to store Ollama process
ollama_process = None

def start_ollama():
    """Start Ollama service"""
    global ollama_process
    try:
        # Check if Ollama is already running
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            st.success("Ollama is already running!")
            return True
    except requests.exceptions.ConnectionError:
        pass

    try:
        # Start Ollama process
        ollama_process = subprocess.Popen(['ollama', 'serve'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        
        # Wait for Ollama to start (max 10 seconds)
        for _ in range(10):
            try:
                response = requests.get('http://localhost:11434/api/tags')
                if response.status_code == 200:
                    st.success("Ollama service started successfully!")
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(1)
                continue
        
        st.error("Failed to start Ollama service. Please start it manually.")
        return False
    except Exception as e:
        st.error(f"Error starting Ollama: {str(e)}")
        return False

def stop_ollama():
    """Stop Ollama service"""
    global ollama_process
    if ollama_process:
        try:
            # Send SIGTERM to the process
            os.kill(ollama_process.pid, signal.SIGTERM)
            ollama_process.wait(timeout=5)
            st.info("Ollama service stopped.")
        except Exception as e:
            st.error(f"Error stopping Ollama: {str(e)}")
            # Force kill if normal termination fails
            try:
                os.kill(ollama_process.pid, signal.SIGKILL)
            except:
                pass

# Register the cleanup function
atexit.register(stop_ollama)

# List of available models
AVAILABLE_MODELS = [
    "llama2",
    "mistral",
    "codellama",
    "neural-chat",
    "starling-lm",
    "dolphin-phi",
    "orca-mini",
    "vicuna",
    "llama2-uncensored",
    "nous-hermes",
    "stable-beluga",
    "wizard-vicuna-uncensored"
]

def get_installed_models():
    """Get list of models installed in Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama. Make sure it's running on port 11434.")
        return []

# Initialize Ollama
def initialize_llm(model_name):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = Ollama(model=model_name, callback_manager=callback_manager)
    return llm

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to generate summary with progress bar
def generate_summary(text, llm):
    # Create a placeholder for the progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate text length for progress simulation
    text_length = len(text)
    chunk_size = max(1, text_length // 100)  # Divide text into 100 chunks
    
    # Initialize progress tracking
    progress = 0
    status_text.text("Starting summarization...")
    
    # Update progress while generating summary
    def progress_callback():
        nonlocal progress
        if progress < 90:  # Cap at 90% until complete
            progress += 1
            progress_bar.progress(progress)
            if progress < 30:
                status_text.text("Analyzing document...")
            elif progress < 60:
                status_text.text("Generating summary...")
            else:
                status_text.text("Finalizing summary...")
    
    # Start progress updates
    progress_timer = time.time()
    
    # Generate the summary
    prompt = f"""Please provide a concise summary of the following text:

{text}

Summary:"""
    
    try:
        summary = llm(prompt)
        # Complete the progress bar
        progress_bar.progress(100)
        status_text.text("Summary complete!")
        return summary
    except Exception as e:
        status_text.text(f"Error generating summary: {str(e)}")
        return None
    finally:
        # Clean up progress elements
        progress_bar.empty()
        status_text.empty()

# Start Ollama service
if not start_ollama():
    st.error("Failed to start Ollama service. Please make sure Ollama is installed and try again.")
    st.stop()

# Streamlit UI
st.title("PDF Summarizer")
st.write("Upload a PDF file to get a summary using local LLM")

# Get installed models
installed_models = get_installed_models()

# Model selection
st.subheader("Model Selection")
if not installed_models:
    st.warning("No models found. Please install at least one model using Ollama.")
    st.info("You can install models using: ollama pull <model_name>")
    st.stop()

# Create two columns for model selection
col1, col2 = st.columns(2)

with col1:
    st.write("Installed Models:")
    selected_model = st.selectbox(
        "Choose a model",
        installed_models,
        key="model_selector"
    )

with col2:
    st.write("Available Models:")
    st.write("To install a new model, run:")
    st.code("ollama pull <model_name>")
    st.write("Available models to install:")
    for model in AVAILABLE_MODELS:
        if model not in installed_models:
            st.write(f"- {model}")

# File uploader
st.subheader("PDF Upload")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Initialize LLM with selected model
    llm = initialize_llm(selected_model)
    
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    
    # Generate summary with progress bar
    summary = generate_summary(text, llm)
    
    if summary:
        # Display summary
        st.subheader("Summary")
        st.write(summary)
        
        # Add a download button for the summary
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

# Add a button to manually stop Ollama
if st.button("Stop Ollama Service"):
    stop_ollama()
    st.stop()

