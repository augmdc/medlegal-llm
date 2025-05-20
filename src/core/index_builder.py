from typing import List, Optional
from llama_index.core import (
    Document,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)
import os
import shutil

class IndexBuilder:
    def __init__(self, persist_root_dir: str = "./llama_index_data"):
        self.persist_root_dir = persist_root_dir
        os.makedirs(self.persist_root_dir, exist_ok=True)

    def _get_storage_context(self, index_name: str) -> StorageContext:
        persist_dir = os.path.join(self.persist_root_dir, index_name)
        return StorageContext.from_defaults(persist_dir=persist_dir)

    def _build_or_load_index(self, index_class, documents: Optional[List[Document]], index_name: str):
        storage_context = self._get_storage_context(index_name)
        index_persist_path = os.path.join(self.persist_root_dir, index_name)

        try:
            if os.path.exists(os.path.join(index_persist_path, "docstore.json")):
                # Attempt to load existing index
                index = load_index_from_storage(storage_context)
                print(f"Loaded existing {index_class.__name__} '{index_name}' from {index_persist_path}")
                if documents: # If new documents are provided, add to existing index
                    print(f"Adding {len(documents)} new documents to existing index '{index_name}'")
                    for doc in documents:
                        index.insert_document(doc) # Assumes nodes are created by the index
                    index.storage_context.persist(persist_dir=index_persist_path)
                    print(f"Persisted updated index '{index_name}' to {index_persist_path}")
                return index
            elif documents:
                # Create new index if it doesn't exist and documents are provided
                print(f"Building new {index_class.__name__} '{index_name}' with {len(documents)} documents.")
                index = index_class.from_documents(documents, storage_context=storage_context)
                index.storage_context.persist(persist_dir=index_persist_path) # Persist after creation
                print(f"Persisted new index '{index_name}' to {index_persist_path}")
                return index
            else:
                # No documents to build new, and no existing index found
                return None
        except Exception as e:
            print(f"Error building/loading {index_class.__name__} '{index_name}': {e}. Consider clearing storage.")
            return None

    def get_vector_index(self, documents: Optional[List[Document]] = None, index_name: str = "vector_index") -> Optional[VectorStoreIndex]:
        return self._build_or_load_index(VectorStoreIndex, documents, index_name)

    def get_summary_index(self, documents: Optional[List[Document]] = None, index_name: str = "summary_index") -> Optional[SummaryIndex]:
        return self._build_or_load_index(SummaryIndex, documents, index_name)

    def clear_index_storage(self, index_name: str):
        index_persist_path = os.path.join(self.persist_root_dir, index_name)
        if os.path.exists(index_persist_path):
            shutil.rmtree(index_persist_path)
            print(f"Cleared storage for index '{index_name}' at {index_persist_path}")
        else:
            print(f"No storage found for index '{index_name}' at {index_persist_path}")

    def clear_all_storage(self):
        if os.path.exists(self.persist_root_dir):
            shutil.rmtree(self.persist_root_dir)
            print(f"Cleared all LlamaIndex storage at {self.persist_root_dir}")
            os.makedirs(self.persist_root_dir, exist_ok=True) # Recreate root after clearing
        else:
            print(f"No LlamaIndex storage found at {self.persist_root_dir}") 