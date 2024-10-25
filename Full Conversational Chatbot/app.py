# import Langchain dependecies
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# import Streamlit for UI
import streamlit as st

# Setup the app
st.title("Ask anything!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# setup the llm
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, max_tokens=200)
template = ChatPromptTemplate.from_template("""
    You are a medical AI chatbot. Users ask you medical questions, and you provide accurate and in a simple way answers based on your training data.

    Prompt: What is the definition of [medical term]?
    How does [medication] work?
    What are the side effects of [treatment]?

    User: {input}
""")

chatbot = template | llm

prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content':prompt})
    response = chatbot.invoke({'input': st.session_state.messages}).content
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({'role': 'assistant', 'content':response})