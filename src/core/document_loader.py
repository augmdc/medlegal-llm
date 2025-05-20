from typing import List
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
import tempfile
import os

class DocumentLoader:
    def __init__(self):
        pass

    def load_pdf(self, pdf_file_content: bytes, filename: str) -> List[Document]:
        """Loads a PDF from bytes, extracts text, and returns a list of LlamaIndex Documents.
           Currently creates one Document object for the entire PDF.
        Args:
            pdf_file_content: Bytes of the PDF file.
            filename: The original name of the file, used for metadata.

        Returns:
            A list of LlamaIndex Document objects extracted from the PDF.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, filename)
            
            # Write PDF content to temporary file
            try:
                with open(temp_file_path, 'wb') as f:
                    f.write(pdf_file_content)
                
                # Use SimpleDirectoryReader to load the PDF
                reader = SimpleDirectoryReader(
                    input_dir=temp_dir,
                    filename_as_id=True
                )
                documents = reader.load_data()
                
                if not documents:
                    print(f"No text extracted from PDF {filename}.")
                    return []
                    
                return documents

            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
                return []