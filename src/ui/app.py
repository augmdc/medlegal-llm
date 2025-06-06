import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from src.utils.ollama_manager import OllamaManager

class MedLegalApp:
    def __init__(self):
        # Initialize the manager. The atexit hook within the manager
        # will handle stopping Ollama when the app process terminates.
        if 'ollama_manager' not in st.session_state:
            st.session_state.ollama_manager = OllamaManager()

        # Initialize session state for app status
        if "ollama_status" not in st.session_state:
            st.session_state.ollama_status = None
        if "installed_models" not in st.session_state:
            st.session_state.installed_models = []

    def setup_ollama(self):
        """Connects to Ollama and gets model info."""
        if st.session_state.ollama_status is None: # Run only once
            with st.spinner("Connecting to Ollama service..."):
                status = st.session_state.ollama_manager.start()
                st.session_state.ollama_status = status
                
                if status in ["already_running", "started"]:
                    st.session_state.installed_models = st.session_state.ollama_manager.get_installed_models()
            st.rerun()

    def render(self):
        st.title("MedLegalLLM App")
        
        # --- Initial Setup ---
        self.setup_ollama()
        
        # --- Sidebar ---
        st.sidebar.title("Configuration")
        
        # Display Ollama Status
        status = st.session_state.ollama_status
        if status == "already_running":
            st.sidebar.success("Connected to existing Ollama service.")
        elif status == "started":
            st.sidebar.success("Ollama service started successfully.")
        elif status == "not_found":
            st.sidebar.error("Ollama not found. Please install it.")
            st.stop()
        elif status == "failed_to_start":
            st.sidebar.error("Failed to start Ollama. Check for conflicts.")
            st.stop()

        # Model Selection Dropdown
        if st.session_state.installed_models:
            st.sidebar.selectbox(
                "Choose LLM Model:",
                st.session_state.installed_models,
                key="llm_model_selector"
            )
        else:
            st.sidebar.warning("No Ollama models found.")

        st.sidebar.markdown("---")
        st.sidebar.title("Service Management")

        if st.sidebar.button("Stop Ollama Service"):
            with st.spinner("Stopping Ollama service..."):
                st.session_state.ollama_manager.stop()
            st.sidebar.success("Ollama service stopped.")
            st.sidebar.info("You can now close this browser tab.")
            st.stop() # Halts the Streamlit script

        st.sidebar.info("Ollama is automatically managed. It will shut down if this app started it and you close the app terminal.")

        # --- Main Page ---
        st.header("1. Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

        if uploaded_file is not None:
            st.info(f"File '{uploaded_file.name}' uploaded.")
            if st.button(f"Process {uploaded_file.name}", key="process_button"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    st.success(f"'{uploaded_file.name}' processed (placeholder).")
            
            st.header("2. Query Document")
            query_text = st.text_input("Ask a question:", key="query_input")
            if query_text and st.button("Search", key="search_button"):
                with st.spinner("Searching..."):
                    st.markdown("This is a placeholder response.")
        else:
            st.info("Upload a PDF to begin.")

def main():
    if 'app' not in st.session_state:
        st.session_state.app = MedLegalApp()
    st.session_state.app.render()

if __name__ == "__main__":
    main()