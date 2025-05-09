from typing import Optional, List
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

class Summarizer:
    def __init__(self, model_name: str = "llama2"):
        self.llm = Ollama(
            model=model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        
        self.rag_prompt = PromptTemplate(
            template="""Use the following context to help summarize the document. 
            If the context is not relevant, you can ignore it.

            Context:
            {context}

            Document to summarize:
            {text}

            Please provide a comprehensive summary that captures the main points and key details.
            Summary:""",
            input_variables=["context", "text"]
        )
        
        self.basic_prompt = PromptTemplate(
            template="""Please provide a concise summary of the following text:

            {text}

            Summary:""",
            input_variables=["text"]
        )
    
    def generate_summary(
        self,
        text: str,
        context: Optional[List[str]] = None,
        progress_callback=None
    ) -> str:
        """Generate a summary of the text, optionally using context."""
        if context:
            # Use RAG prompt with context
            prompt = self.rag_prompt.format(
                context="\n\n".join(context),
                text=text
            )
        else:
            # Use basic prompt without context
            prompt = self.basic_prompt.format(text=text)
        
        return self.llm(prompt) 