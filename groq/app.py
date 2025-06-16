import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
import time


load_dotenv()

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceHubEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.document = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.document)

    st.session_state.vector = Chroma.from_documents(st.session_state.documents, st.session_state.embeddings)

st.title("ChatGroq Demo")
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7)
prompt = ChatPromptTemplate.from_template("""
You are an helpful AI.
Answer the following questions based on the provided context.
<context>
{context}
</context>
Question: {input}                      
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriver = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriver, document_chain)

prompt = st.text_input("Input Your Prompt Here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    process_time = time.process_time() - start
    st.write(response["answer"])
    st.write(f"Process Time: {process_time:.2f} seconds")
