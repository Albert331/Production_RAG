from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def ingest_pdf(file_bytes: bytes, filename: str, vector_store) -> int:
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        
        for chunk in chunks:
            chunk.metadata['source'] = filename

        vector_store.add_documents(chunks)
        return len(chunks)

    finally:
        os.unlink(tmp_path)