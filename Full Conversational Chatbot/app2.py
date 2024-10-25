# app.py
import streamlit as st
from pdf_loader import load_pdf
from model_setup import initialize_llm
from chat_chain import create_retrieval_chain

# Load the PDF database
@st.cache_resource
def load_parameters():
    db =  load_pdf('Java Full Notes.pdf') 
    llm = initialize_llm()
    chain = create_retrieval_chain(llm, db.as_retriever())

    return chain

try:
    chain = load_parameters()

except Exception as e:
    st.error(f"Error setting up retrieval chain: {str(e)}")
    st.stop()

# Set up the Streamlit app title and chat session
st.title("Ask anything!")
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Get user input and process the response
prompt = st.chat_input("Ask a question about the Java notes...")
if prompt:
    try:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message("user").markdown(prompt)
        
        # Retrieve response with source documents if relevant
        response = chain({
            "question": prompt, 
            "chat_history": [(msg['role'], msg['content']) for msg in st.session_state.messages]
        })
        answer = response['answer']
        
        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({'role': 'assistant', 'content': answer})

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
