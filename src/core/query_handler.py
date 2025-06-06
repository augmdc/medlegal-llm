from typing import Optional
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.query_engine import QueryEngine, RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

class QueryHandler:
    def get_vector_query_engine(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 3,
        use_hybrid_search: bool = True
    ) -> Optional[QueryEngine]:
        """Creates a query engine for a VectorStoreIndex.
        If use_hybrid_search is True, it currently uses vector search with a doubled similarity_top_k.
        """
        if not index:
            print("Error: Index not provided to get_vector_query_engine.")
            return None
            
        effective_top_k = similarity_top_k
        if use_hybrid_search:
            # The original "hybrid" mode effectively increased top_k for the vector retriever.
            # True hybrid search (e.g., BM25 + vector) was not fully implemented.
            # This maintains the behavior of fetching more results from the vector store.
            effective_top_k = similarity_top_k * 2
            # Consider logging or print statement if detailed clarification of "hybrid" mode is needed at runtime.
            # print("Note: Hybrid search enabled. Using vector search with increased top_k.")

        try:
            retriever = index.as_retriever(similarity_top_k=effective_top_k)
        except Exception as e:
            print(f"Failed to create vector retriever: {e}. Falling back to default similarity_top_k.")
            try:
                retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            except Exception as e_fallback:
                print(f"Failed to create vector retriever even with default top_k: {e_fallback}.")
                return None
        
        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]
        )

    def get_summary_query_engine(
        self,
        index: SummaryIndex,
        response_mode: str = "tree_summarize"
    ) -> Optional[QueryEngine]:
        """Creates a query engine for a SummaryIndex."""
        if not index:
            print("Error: Index not provided to get_summary_query_engine.")
            return None
        try:
            return index.as_query_engine(response_mode=response_mode)
        except Exception as e:
            print(f"Failed to create summary query engine: {e}.")
            return None

    def query_index(self, query_engine: Optional[QueryEngine], query_text: str) -> Optional[str]:
        """Executes a query using the provided query engine."""
        if not query_engine:
            return "Query engine not available."
        if not query_text or not query_text.strip():
            return "Query text is empty."
        try:
            response = query_engine.query(query_text)
            return str(response)
        except Exception as e:
            return f"Error during query: {e}" 