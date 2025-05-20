from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

def configure_llama_index_settings(
    llm_model_name: str = "llama2",
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """Configures global LlamaIndex settings for LLM, embedding model, and node parser."""
    Settings.llm = Ollama(model=llm_model_name)
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"LlamaIndex global settings configured: LLM='{llm_model_name}', EmbedModel='{embedding_model_name}', ChunkSize={chunk_size}") 