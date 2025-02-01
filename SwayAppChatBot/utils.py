# utils.py
from langchain.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

def create_retriever_tool_agent(pdf_path: str) -> Tool:
    """Create a retrieval tool from PDF documents."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    retriever = vectorstore.as_retriever()
    
    return Tool(
        name="pdf_retriever",
        func=retriever.get_relevant_documents,
        description="Useful for retrieving relevant information from the uploaded PDF document."
    )

def get_prompt() -> str:
    """Return the system prompt for the chat agent."""
    return ChatPromptTemplate.from_messages([("system", """You are a helpful academic assistant. When asked questions, please:
    1. Use the PDF retriever tool if available to find relevant information
    2. Provide clear, concise answers with citations where appropriate
    3. If you're unsure about something, admit it and suggest alternatives
    4. Keep responses focused on academic content and student support"""), 
    ("human", "{input} {agent_scratchpad}")
    ])