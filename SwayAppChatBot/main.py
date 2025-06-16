# main.py
import streamlit as st
from pathlib import Path
from typing import Optional
import logging
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from logger import setup_logger
from utils import create_retriever_tool_agent, get_prompt
from dotenv import load_dotenv
load_dotenv()

# Initialize settings and logger
logger = setup_logger(__name__)

class StudentHelplineApp:
    def __init__(self):
        self.setup_streamlit()
        self.initialize_state()
        
    def setup_streamlit(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="24X7 Student Helpline",
            page_icon="📚",
            # layout="wide"
        )
        st.title("24X7 Helpline Support For Students")
        
    def initialize_state(self):
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'agent_executor' not in st.session_state:
            st.session_state.agent_executor = None
        if 'uploaded_pdf_name' not in st.session_state:
            st.session_state.uploaded_pdf_name = set()
            
    def create_agent(self, uploaded_pdf = None) -> AgentExecutor:
        """Create an agent with optional PDF tool."""
        try:
            tools = []
            if uploaded_pdf:
                pdf_path = self.save_uploaded_pdf(uploaded_pdf)
                pdf_retriever_tool = create_retriever_tool_agent(pdf_path)
                tools.append(pdf_retriever_tool)
                
            llm = ChatGroq(
                model="llama3-8b-8192",
                temperature=0.5,
                max_tokens=512
            )
            
            agent = create_tool_calling_agent(
                llm=llm,
                tools=tools,
                prompt=get_prompt()
            )
            
            return AgentExecutor(agent=agent, tools=tools, verbose=True)
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            st.error("Failed to initialize the assistant. Please try again.")
            raise
            
    def save_uploaded_pdf(self, uploaded_pdf) -> str:
        """Save uploaded PDF and return its path."""
        try:
            pdf_dir = Path("pdfs")
            pdf_dir.mkdir(exist_ok=True)
            
            pdf_path = pdf_dir / uploaded_pdf.name
            pdf_path.write_bytes(uploaded_pdf.getvalue())
            
            logger.info(f"PDF saved successfully: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logger.error(f"Error saving PDF: {str(e)}")
            st.error("Failed to save the PDF. Please try again.")
            raise
            
    def generate_response(self, prompt: str) -> str:
        """Generate response using the agent."""
        try:
            response = st.session_state.agent_executor.invoke(
                {"input": prompt}
            )["output"]
            logger.info(f"Generated response for prompt: {prompt[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error. Please try asking your question again."
            
    def run(self):
        """Run the main application loop."""
        try:
            # Display chat history at the beginning
            for message in st.session_state.messages[-5:]:
                st.chat_message(message['role']).markdown(message['content'])

            # File uploader
            uploaded_pdf = st.file_uploader(
                "Upload a PDF document for reference",
                type="pdf",
                help="Upload your study materials or course documents"
            )
            
            # Initialize or update agent if needed
            if not uploaded_pdf and not st.session_state.agent_executor:
                st.session_state.agent_executor = self.create_agent(uploaded_pdf)

            if uploaded_pdf and (uploaded_pdf.name not in st.session_state.uploaded_pdf_name):
                st.session_state.agent_executor = self.create_agent(uploaded_pdf)
                st.session_state.uploaded_pdf_name.add(uploaded_pdf.name)

            # Chat interface
            if prompt := st.chat_input("Ask any question related to your curriculum"):
                # Add message to history
                st.session_state.messages.append({'role': 'user', 'content': prompt})
                
                # Generate response
                response = self.generate_response(prompt)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                
                # Rerun to update the chat display
                st.rerun()
                
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    app = StudentHelplineApp()
    app.run()