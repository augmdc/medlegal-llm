import streamlit as st
from typing import Optional
import os
import psutil

from src.core.global_settings import configure_llama_index_settings
from src.core.document_loader import DocumentLoader
from src.core.index_builder import IndexBuilder
from src.core.query_handler import QueryHandler
from src.utils.ollama_manager import OllamaManager

# Initialize LlamaIndex settings early
# These can be pulled from a config file or UI in a more complex app
LLM_MODEL = "llama2" # Default, can be changed via UI
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # Default
configure_llama_index_settings(
    llm_model_name=LLM_MODEL,
    embedding_model_name=EMBEDDING_MODEL
)

class PDFQueryApp:
    def __init__(self):
        self.ollama_manager = OllamaManager()
        self.doc_loader = DocumentLoader()
        self.index_builder = IndexBuilder() # Default persist_dir = ./llama_index_data
        self.query_handler = QueryHandler()

        # Initialize session state keys
        default_session_state = {
            "vector_index": None,
            "summary_index": None,
            "uploaded_filename": None,
            "ollama_status_checked": False,
            "selected_llm_model": LLM_MODEL,
            "installed_models": [],
            "document_processed": False,
        }
        for key, value in default_session_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def cleanup_on_exit(self):
        print("Performing cleanup...")
        self.ollama_manager.stop() # Stop Ollama if managed by app
        # Optionally clear all LlamaIndex data on exit, or manage through UI
        # self.index_builder.clear_all_storage()
        # print("LlamaIndex storage cleared.")
        # Clear session state explicitly
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        print("Session state cleared.")


    def stop_application(self):
        self.cleanup_on_exit()
        current_process = psutil.Process(os.getpid())
        parent = current_process.parent()
        if parent:
            parent.terminate()
        os._exit(0)

    def setup_ollama(self):
        if not st.session_state.ollama_status_checked:
            with st.spinner("Connecting to Ollama service..."):
                ollama_status = self.ollama_manager.start()
            
            if ollama_status == "already_running":
                st.success("Successfully connected to existing Ollama service.")
            elif ollama_status == "started":
                st.success("Ollama service started successfully by the app.")
            elif ollama_status == "not_found":
                st.error("Ollama command not found. Ensure Ollama is installed and in PATH.")
                st.info("Download Ollama from https://ollama.com")
                st.stop()
            elif ollama_status == "failed_to_start":
                st.error("Failed to start/connect to Ollama. Check installation/conflicts.")
                st.stop()
            
            st.session_state.installed_models = self.ollama_manager.get_installed_models()
            if not st.session_state.installed_models:
                st.warning("No Ollama models found. Please install models, e.g., `ollama pull llama2`")
            else:
                # Update default if current selection not installed
                if st.session_state.selected_llm_model not in st.session_state.installed_models:
                    st.session_state.selected_llm_model = st.session_state.installed_models[0]
            
            st.session_state.ollama_status_checked = True
            st.rerun() # Rerun to reflect model list and status

    def render_ui(self):
        st.title("MedLegalLLM App")

        self.setup_ollama()

        st.sidebar.title("Configuration")
        
        # Model Selection
        if st.session_state.installed_models:
            new_selected_llm = st.sidebar.selectbox(
                "Choose LLM Model (for Querying/Summarization):",
                st.session_state.installed_models,
                index=st.session_state.installed_models.index(st.session_state.selected_llm_model)
                    if st.session_state.selected_llm_model in st.session_state.installed_models else 0,
                key="llm_model_selector"
            )
            if new_selected_llm != st.session_state.selected_llm_model:
                st.session_state.selected_llm_model = new_selected_llm
                # Reconfigure LlamaIndex settings if model changes
                with st.spinner(f"Switching to LLM: {new_selected_llm}..."):
                    configure_llama_index_settings(llm_model_name=new_selected_llm, embedding_model_name=EMBEDDING_MODEL)
                st.rerun()
        else:
            st.sidebar.warning("No Ollama models available for selection.")

        st.sidebar.markdown("---")
        st.sidebar.title("Database Management")
        if st.sidebar.button("Clear All Indexed Data"):
            with st.spinner("Clearing all LlamaIndex data..."):
                self.index_builder.clear_all_storage()
            st.session_state.vector_index = None
            st.session_state.summary_index = None
            st.session_state.document_processed = False
            st.session_state.uploaded_filename = None
            st.sidebar.success("All indexed data cleared!")
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.title("Service Management")
        if st.sidebar.button("Stop Application & Clean Up"):
            st.sidebar.warning("Stopping application...")
            self.stop_application()

        # Main page content
        st.header("1. Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

        if uploaded_file is not None:
            if st.session_state.uploaded_filename != uploaded_file.name:
                st.session_state.document_processed = False # New file, reset processed flag

            if not st.session_state.document_processed:
                if st.button(f"Process {uploaded_file.name}", key="process_button"):
                    with st.spinner(f"Loading and Processing {uploaded_file.name}..."):
                        pdf_bytes = uploaded_file.getvalue()
                        llama_documents = self.doc_loader.load_pdf(pdf_bytes, uploaded_file.name)
                        
                        if llama_documents:
                            # Build/Load Vector Index
                            vec_index = self.index_builder.get_vector_index(llama_documents)
                            if vec_index:
                                st.session_state.vector_index = vec_index
                                st.success(f"Vector Index ready for '{uploaded_file.name}'.")
                            else:
                                st.error("Failed to build/load Vector Index.")

                            # Build/Load Summary Index
                            sum_index = self.index_builder.get_summary_index(llama_documents)
                            if sum_index:
                                st.session_state.summary_index = sum_index
                                st.success(f"Summary Index ready for '{uploaded_file.name}'.")
                            else:
                                st.error("Failed to build/load Summary Index.")
                            
                            if vec_index or sum_index:
                                st.session_state.uploaded_filename = uploaded_file.name
                                st.session_state.document_processed = True
                        else:
                            st.error(f"Could not extract any documents from {uploaded_file.name}.")
                    st.rerun() # Rerun to update UI based on processed state
            
            if st.session_state.document_processed and st.session_state.uploaded_filename == uploaded_file.name:
                st.info(f"Document '{st.session_state.uploaded_filename}' is processed and indexed.")
                
                # Querying Section
                st.header("2. Query Document (Vector Search)")
                if st.session_state.vector_index:
                    query_text_vector = st.text_input("Ask a question about the document (vector search):", key="vector_query")
                    use_hybrid = st.checkbox("Use Hybrid Search (Vector + Keyword)", value=True, key="hybrid_cb")
                    
                    if query_text_vector and st.button("Search Document", key="search_doc_btn"):
                        with st.spinner("Searching document..."):
                            query_engine = self.query_handler.get_vector_query_engine(
                                st.session_state.vector_index, 
                                use_hybrid_search=use_hybrid
                            )
                            response = self.query_handler.query_index(query_engine, query_text_vector)
                            st.markdown("#### Response:")
                            st.markdown(response if response else "No response or error.")
                else:
                    st.warning("Vector Index not available for querying.")

                # Summarization Section
                st.header("3. Get Summary (Summary Index)")
                if st.session_state.summary_index:
                    summary_query_text = "Please provide a comprehensive summary of the document." # Default query
                    custom_summary_prompt = st.text_area(
                        "Or, enter a custom prompt for summarization:", 
                        value=summary_query_text,
                        key="summary_prompt_input"
                    )

                    if st.button("Generate Summary", key="summarize_doc_btn"):
                        with st.spinner("Generating summary..."):
                            query_engine = self.query_handler.get_summary_query_engine(st.session_state.summary_index)
                            response = self.query_handler.query_index(query_engine, custom_summary_prompt)
                            st.markdown("#### Summary:")
                            st.markdown(response if response else "No summary or error.")
                else:
                    st.warning("Summary Index not available for summarization.")
        else:
            st.info("Upload a PDF to begin.")
            # Reset if file is removed
            if st.session_state.uploaded_filename: 
                st.session_state.vector_index = None
                st.session_state.summary_index = None
                st.session_state.document_processed = False
                st.session_state.uploaded_filename = None

    def run(self):
        self.render_ui()