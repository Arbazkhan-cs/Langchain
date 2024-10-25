# pdf_loader.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def load_pdf(pdf_path):
    """
    Load a PDF, split it into chunks with embeddings, and create a Chroma vector store for retrieval.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        db: Chroma vector store database for retrieval.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ";"],
        length_function=len
    )
    documents = splitter.split_documents(document)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    db = Chroma.from_documents(documents, embeddings)
    return db
