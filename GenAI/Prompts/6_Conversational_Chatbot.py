# import neccessary libraires
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import streamlit as st

# load environment varaibles
load_dotenv()

# initialize model
model = ChatGroq(model="llama3-8b-8192", temperature=0.6)

# Define Chat Prompt Template
chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are specialist AI teacher. You talks very politely and answers to the user question in very simple way"),
    MessagesPlaceholder(variable_name="input_text")
])

# create chain
chat_chain = chat_prompt_template | model

# ========================= Design UI ================================================
st.title("Conversational Chatbot!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi, how can i assist you today?")
    ]

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Handle user input
if prompt := st.chat_input("Enter your message here..."):
    # Add user message to chat history
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("human"):
        st.markdown(prompt)

    # Generate and display AI response with streaming
    with st.chat_message("assistant"):
        
        chat_history = st.session_state.messages[:-1]
        input_text = [HumanMessage(content=f"Chat history: {chat_history}\n\nCurrent input: {prompt}")]

        print(input_text)

        # Stream the response
        stream = chat_chain.stream({
            "input_text": input_text
        })
        
        response = st.write_stream(stream)
    
    # Add AI response to chat history
    ai_message = AIMessage(content=response)
    st.session_state.messages.append(ai_message)