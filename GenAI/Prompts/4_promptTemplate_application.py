# Import neccessary libraris
from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st

# Load Environment Varaible
load_dotenv()

# Initialize model
model = ChatGroq(model="llama3-8b-8192", temperature=0.6)

# Define PromptTemplate
path = "research_paper_prompt_template.json"
prompt_template = load_prompt(path)

# create chain
chat_chain = prompt_template | model

# Design UI 
st.title("Research Paper Summarizer")

col1, col2, col3 = st.columns(3)

with col1:
    paper_title = st.radio(
        "Research paper:",
        ["Attention is all you need", "BERT", "Deep Q-Networks", "AlphaFold"],
        key="paper_title"
    )

with col2:
    input_style = st.radio(
        "Explanation style:",
        ["Simple", "Complex", "Mathmatical"],
        key="input_style"
    )

with col3:
    input_length = st.radio(
        "Explanation Length:",
        ["Short", "Medium", "Long"],
        key="input_length"
    )

if st.button("SUBMIT"):
    st.write("Generating summary...") 
    result = chat_chain.invoke({
        "paper_title": paper_title,
        "input_style": input_style,
        "input_length": input_length
    })

    st.subheader("Summary:")
    st.write(result.content)