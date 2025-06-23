# import necessary libraries
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load/create documents
docs = [
    Document(
        page_content="LangChain is a powerful framework designed to build applications with language models. It allows chaining together LLMs and tools like retrievers, APIs, and memory to create dynamic, multi-step workflows.",
        metadata={"topic": "LangChain", "type": "framework", "language": "Python"}
    ),
    Document(
        page_content="Pandas is an open-source Python library that provides data structures and data analysis tools. It is widely used for data cleaning, manipulation, and analysis in machine learning pipelines.",
        metadata={"topic": "Pandas", "type": "library", "language": "Python"}
    ),
    Document(
        page_content="PyTorch is a deep learning framework developed by Meta AI. It offers dynamic computation graphs, making it highly flexible for research and production. It supports GPUs and is commonly used in academia and industry.",
        metadata={"topic": "PyTorch", "type": "framework", "language": "Python"}
    ),
    Document(
        page_content="LangChain enables the use of LLMs in pipelines where they can make decisions, access tools like Pandas for data analysis, and interact with other Python modules like PyTorch for model predictions.",
        metadata={"topic": "LangChain + Pandas + PyTorch", "type": "integration", "use_case": "AI Pipelines"}
    ),
    Document(
        page_content="You can integrate LangChain with Pandas to process user-uploaded CSV files, and use PyTorch models behind the scenes to perform predictions, making it ideal for intelligent data apps.",
        metadata={"topic": "LangChain + PyTorch + Pandas", "type": "use_case", "application": "intelligent apps"}
    )
]

# create vector store with document
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding
)

# create retriver
retriever = vectorstore.as_retriever(
    search_kwargs={"k":3, "lambda_mult": 0.5},
    search_type="mmr"
)

# test the retriever
query = "which among is the best library for agentic workflow"
result = retriever.invoke(query)

# print the result
for doc in result:
    print(f"\n---\nContent:\n{doc.page_content}\nMetadata:\n{doc.metadata}")
