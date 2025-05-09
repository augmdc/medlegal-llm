import streamlit as st
from typing import Optional
import time
import sys
import signal
import os
import psutil

from ..core.document_processor import DocumentProcessor
from ..core.summarizer import Summarizer
from ..db.vector_store import VectorStore
from ..utils.ollama_manager import OllamaManager

class PDFSummarizerApp:
    def __init__(self):
        self.ollama_manager = OllamaManager()
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        # Initialize session state
        if 'current_summary' not in st.session_state:
            st.session_state.current_summary = None
        if 'current_filename' not in st.session_state:
            st.session_state.current_filename = None
        if 'should_stop' not in st.session_state:
            st.session_state.should_stop = False
    
    def cleanup(self):
        """Perform cleanup operations before stopping the app."""
        try:
            # Stop Ollama service
            self.ollama_manager.stop()
            
            # Clear vector store
            self.vector_store.clear()
            
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            return True
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")
            return False
    
    def stop_application(self):
        """Stop the application forcefully."""
        if self.cleanup():
            # Get the current process
            current_process = psutil.Process(os.getpid())
            # Get the parent process (the terminal process)
            parent = current_process.parent()
            if parent:
                # Send SIGTERM to the parent process
                parent.terminate()
            # Force exit the current process
            os._exit(0)
    
    def setup_ui(self):
        """Setup the Streamlit UI."""
        # Check if we should stop
        if st.session_state.should_stop:
            st.stop()
        
        st.title("PDF Summarizer with RAG")
        st.write("Upload a PDF file to get a summary using local LLM and RAG")
        
        # Attempt to start/connect to Ollama service
        with st.spinner("Connecting to Ollama service..."):
            ollama_status = self.ollama_manager.start()

        if ollama_status == "already_running":
            st.success("Successfully connected to existing Ollama service.")
        elif ollama_status == "started":
            st.success("Ollama service started successfully by the app.")
        elif ollama_status == "not_found":
            st.error("Ollama command not found. Please ensure Ollama is installed and in your system's PATH.")
            st.info("You can download Ollama from https://ollama.com")
            st.stop()
        elif ollama_status == "failed_to_start":
            st.error("Failed to start or connect to Ollama service. Please check if Ollama is installed correctly and no other instance is conflicting or port 11434 is in use.")
            st.stop()
        
        # Get installed models
        installed_models = self.ollama_manager.get_installed_models()
        
        # Model selection UI
        st.subheader("Model Selection")
        if not installed_models:
            st.warning("No Ollama models found. Please install at least one model.")
            st.info("You can pull models using: ollama pull <model_name> (e.g., ollama pull llama2)")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Embedding Model:")
            self.selected_embedding_model = st.selectbox(
                "Choose an embedding model",
                installed_models if installed_models else ["No models installed"],
                key="embedding_model_selector",
                disabled=not installed_models
            )
            st.caption("Used for document vectorization (semantic search)")
        
        with col2:
            st.write("Chat Model:")
            self.selected_chat_model = st.selectbox(
                "Choose a chat model for summarization",
                installed_models if installed_models else ["No models installed"],
                key="chat_model_selector",
                disabled=not installed_models
            )
            st.caption("Used for generating summaries")
    
    def process_document(self, uploaded_file) -> Optional[str]:
        """Process the uploaded document and generate summary."""
        if not self.selected_embedding_model or "No models installed" in self.selected_embedding_model or \
           not self.selected_chat_model or "No models installed" in self.selected_chat_model:
            st.error("Please ensure valid embedding and chat models are selected and installed.")
            return None
            
        summarizer = Summarizer(model_name=self.selected_chat_model)
        self.vector_store.update_embedding_model(self.selected_embedding_model)
        
        with st.spinner("Extracting text from PDF..."):
            text, chunks = self.document_processor.process_document(uploaded_file)
        
        with st.spinner(f"Embedding document using {self.selected_embedding_model}..."):
            num_chunks = self.vector_store.add_documents(
                chunks,
                metadata={"source": uploaded_file.name}
            )
            st.info(f"Document processed into {num_chunks} chunks and embedded.")
        
        use_rag = st.checkbox("Use RAG for enhanced summarization (requires relevant documents in DB)", value=True)
        
        summary_text = f"Generating summary using {self.selected_chat_model}"
        if use_rag:
            summary_text += " with RAG..."
        else:
            summary_text += "..."

        with st.spinner(summary_text):
            context_to_use = None
            if use_rag:
                with st.spinner(f"Searching for relevant context with {self.selected_embedding_model}..."):
                    similar_docs = self.vector_store.search_similar(text)
                    if similar_docs:
                        context_to_use = [doc.page_content for doc in similar_docs]
                        st.info(f"Retrieved {len(context_to_use)} relevant context snippets.")
                    else:
                        st.info("No relevant context found in the vector store for RAG.")
            
            summary = summarizer.generate_summary(text, context=context_to_use)
            
            # Store summary and filename in session state
            st.session_state.current_summary = summary
            st.session_state.current_filename = uploaded_file.name
        
        return summary
    
    def run(self):
        """Run the Streamlit app."""
        self.setup_ui()
        
        st.subheader("PDF Upload & Summarization")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
        if uploaded_file is not None:
            if st.button("Summarize PDF", key="summarize_button"):
                summary_result = self.process_document(uploaded_file)
                
                if summary_result:
                    st.subheader("Generated Summary")
                    st.markdown(summary_result)
                    st.download_button(
                        label="Download Summary",
                        data=summary_result,
                        file_name=f"{uploaded_file.name}_summary.txt",
                        mime="text/plain",
                        key="download_summary"
                    )
        
        # Display current summary if it exists in session state
        if st.session_state.current_summary:
            st.subheader("Current Summary")
            st.markdown(st.session_state.current_summary)
            st.download_button(
                label="Download Current Summary",
                data=st.session_state.current_summary,
                file_name=f"{st.session_state.current_filename}_summary.txt",
                mime="text/plain",
                key="download_current_summary"
            )
        
        st.sidebar.title("Database Management")
        if st.sidebar.button("Clear Vector Database"):
            self.vector_store.clear()
            st.sidebar.success("Vector database cleared!")
            st.experimental_rerun()

        st.sidebar.title("Service Management")
        if st.sidebar.button("Stop Application"):
            st.sidebar.warning("Stopping application...")
            self.stop_application()

if __name__ == "__main__":
    app = PDFSummarizerApp()
    app.run() 