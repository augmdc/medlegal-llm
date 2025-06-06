"""
This script is used to test the LlamaIndex library.
It loads a document, indexes it, and then queries it.
It uses the ChromaVectorStore to store the indexed documents.
It uses the SimpleDirectoryReader to load the documents.
It uses the VectorStoreIndex to index the documents.
It uses the query_engine to query the indexed documents.
It uses the response_mode to keep the answers tight.
It uses the similarity_top_k to set the number of top results to return.
"""

import os
os.environ["OLLAMA_HOST"] = "http://localhost:11434"

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from pathlib import Path

# configure models
Settings.llm = Ollama(model="llama2")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

pdf_path = Path("/Users/augmdc/Downloads/Recents/A review of machine learning approaches for electric vehicle energy consumption modelling in urban transportation.pdf")
if not pdf_path.exists():
    print(f"Error: Document not found at {pdf_path}")
    exit()

# Loading
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("test_collection")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Storing is handled by ChromaDB's persistent client.

# Querying
query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"   # keeps answers tight
)

response = query_engine.query("Give me a summary of the document")
print(response)

# Retrieving


# Evaluation – use the LlamaIndex evaluation module to create ground-truth Q-A pairs and compute retrieval+answer quality.


# Output
