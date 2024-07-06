import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import time

load_dotenv()

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceHubEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader = PyPDFDirectoryLoader("./us_census")
    st.session_state.document = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.document)

    st.session_state.vector = Chroma.from_documents(st.session_state.documents, st.session_state.embeddings)

st.title("HuggingFace Demo")
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"temperature":0.7, "max_length":500})

prompt_template="""
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context
<context>
{context}
</context>

Question:{question}

Answer:  
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=st.session_state.vector.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)

query = st.text_input("Enter your prompt here")

if query:
    start = time.process_time()
    response = retrievalQA.invoke({'query': query})
    process_time = time.process_time() - start
    st.write(response["result"])
    st.write(f"Process time: {process_time} seconds")
