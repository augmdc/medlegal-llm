from typing import Optional
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    QueryEngine
)
from llama_index.retrievers import BM25Retriever, VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor

class QueryHandler:
    def __init__(self):
        # LLM and EmbedModel are configured globally via Settings
        pass

    def get_vector_query_engine(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 3,
        use_hybrid_search: bool = True
    ) -> Optional[QueryEngine]:
        """Creates a query engine for a VectorStoreIndex, optionally with hybrid search."""
        if not index:
            return None
            
        if use_hybrid_search:
            try:
                vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
                # BM25Retriever requires nodes to be in the docstore
                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=index.docstore,
                    similarity_top_k=similarity_top_k
                )
                # Combine both retrievers
                retrievers = [vector_retriever, bm25_retriever]
                combined_retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=similarity_top_k * len(retrievers)
                )
            except Exception as e:
                print(f"Failed to create hybrid retriever: {e}. Falling back to vector retriever.")
                # Fallback to simple vector retriever
                combined_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        else:
            combined_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        
        return RetrieverQueryEngine.from_args(
            retriever=combined_retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )

    def get_summary_query_engine(
        self,
        index: SummaryIndex,
        response_mode: str = "tree_summarize"
    ) -> Optional[QueryEngine]:
        """Creates a query engine for a SummaryIndex."""
        if not index:
            return None
        return index.as_query_engine(response_mode=response_mode)

    def query_index(self, query_engine: Optional[QueryEngine], query_text: str) -> Optional[str]:
        """Executes a query using the provided query engine."""
        if not query_engine:
            return "Query engine not available."
        try:
            response = query_engine.query(query_text)
            return str(response)
        except Exception as e:
            return f"Error during query: {e}" 