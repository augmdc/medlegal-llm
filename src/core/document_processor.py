from typing import Dict, List, Tuple
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from a PDF file."""
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.text_splitter.split_text(text)
    
    def process_document(self, pdf_file) -> Tuple[str, List[str]]:
        """Process a PDF document and return both full text and chunks."""
        text = self.extract_text_from_pdf(pdf_file)
        chunks = self.split_text(text)
        return text, chunks 