from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os

# step - 1: Load the model
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true" # for langsmith tracing
llm = ChatGroq(model="llama3-8b-8192", temperature=0.8, max_tokens=512)

# step- 2: Prompt template
template=[("system", "You are a Helpfull AI. Please response to the user queries"),
        ("user", "Question:{question}")]
prompt = ChatPromptTemplate.from_messages(template)

# step - 4: Chains and output parser
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit setup
st.title("Langchin demo with GROQ Llama3-8b-8192 model")
input_text = st.text_input("Search the topic you want")

if input_text:
    st.write(chain.invoke({"question": input_text}))
else:
    st.write("Please enter the text to search")
