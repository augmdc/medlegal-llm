from typing import List, Dict, Any
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
import time

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self._store = None
        self._embeddings = None
        self._current_model = None
    
    def update_embedding_model(self, model_name: str) -> None:
        """Update the embedding model."""
        if self._current_model != model_name:
            self._current_model = model_name
            self._embeddings = OllamaEmbeddings(model=model_name)
            # Reinitialize store with new embeddings
            self._store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._embeddings
            )
    
    @property
    def store(self) -> Chroma:
        """Lazy initialization of the vector store."""
        if self._store is None:
            # Default to llama2 if no model is selected
            model_name = self._current_model or "llama2"
            self._embeddings = OllamaEmbeddings(model=model_name)
            self._store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._embeddings
            )
        return self._store
    
    def add_documents(self, texts: List[str], metadata: Dict[str, Any]) -> int:
        """Add documents to the vector store."""
        # Add timestamp to metadata
        metadata["timestamp"] = time.time()
        
        # Add documents to vector store
        self.store.add_texts(
            texts=texts,
            metadatas=[metadata] * len(texts)
        )
        
        # Persist the database
        self.store.persist()
        
        return len(texts)
    
    def search_similar(self, query: str, k: int = 3) -> List[Any]:
        """Search for similar documents."""
        return self.store.similarity_search(query, k=k)
    
    def clear(self) -> None:
        """Clear the vector store."""
        if self._store is not None:
            self._store.delete_collection()
            self._store = None
            self._embeddings = None
            self._current_model = None 