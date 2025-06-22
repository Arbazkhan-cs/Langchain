from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import streamlit as st
import os
import validators
import time
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Web Q&A ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_model():
    """Initialize the ChatGroq model"""
    try:
        model = ChatGroq(
            model="llama3-8b-8192", 
            temperature=0.7,
            max_tokens=1000
        )
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def create_prompt_template():
    """Create the prompt template"""
    return PromptTemplate(
        template="""You are a helpful AI assistant that answers questions based on web content.
        
        Question: {question}
        
        Web Content: {data}
        
        Instructions:
        - Provide a clear, accurate answer based on the web content
        - If the information is not available in the content, mention that
        - Be concise but comprehensive
        - Use bullet points when listing multiple items
        
        Answer:""",
        input_variables=["question", "data"]
    )

def extract_data(inputs):
    """Extract data from URL with better error handling"""
    try:
        url = inputs["url"]
        question = inputs["question"]
        
        # Validate URL
        if not validators.url(url):
            return {"question": question, "data": "Invalid URL format. Please provide a valid URL."}
        
        # Load web content
        with st.spinner("Loading web content..."):
            loader = WebBaseLoader(web_path=url)
            documents = loader.load()
            
            if not documents:
                return {"question": question, "data": "No content found at the provided URL."}
            
            content = documents[0].page_content
            
            # # Limit content length to avoid token limits
            # if len(content) > 10000:
            #     content = content[:10000] + "... (content truncated)"
            
            return {"question": question, "data": content}
            
    except Exception as e:
        error_msg = f"Error loading URL: {str(e)}"
        return {"question": inputs.get("question", ""), "data": error_msg}

def main():
    # Header
    st.markdown("<h1 class='main-header'>🤖 Web Q&A ChatBot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Ask questions about any webpage content</p>", unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, help="Controls randomness in responses")
        max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100, help="Maximum response length")
        
        # API Key status
        st.subheader("API Status")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            st.success("✅ GROQ API Key loaded")
        else:
            st.error("❌ GROQ API Key not found")
            st.info("Please set your GROQ_API_KEY in the .env file")
        
        # Instructions
        st.subheader("📖 How to Use")
        st.markdown("""
        1. Enter a valid webpage URL
        2. Type your question about the content
        3. Click 'Get Answer' to analyze
        4. View the AI-generated response
        """)
        
        # Example URLs
        st.subheader("📝 Example URLs")
        example_urls = [
            "https://www.wikipedia.org/wiki/Artificial_intelligence",
            "https://news.ycombinator.com/",
            "https://www.github.com/",
        ]
        for url in example_urls:
            if st.button(f"Use: {urlparse(url).netloc}", key=url):
                st.session_state.example_url = url
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # URL input
        url_input = st.text_input(
            "🔗 Enter Website URL:",
            value=st.session_state.get('example_url', ''),
            placeholder="https://example.com",
            help="Enter the complete URL including https://"
        )
        
        # Question input
        question_input = st.text_area(
            "❓ Enter Your Question:",
            placeholder="What is this webpage about? What are the main features? etc.",
            height=100,
            help="Ask specific questions about the webpage content"
        )
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        get_answer_btn = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.session_state.clear()
            st.rerun()
    
    # Process the request
    if get_answer_btn:
        if not url_input or not question_input:
            st.warning("⚠️ Please provide both URL and question!")
        elif not groq_api_key:
            st.error("❌ GROQ API Key is required. Please check your .env file.")
        else:
            # Initialize components
            model = initialize_model()
            if not model:
                st.error("Failed to initialize the model. Please check your API key.")
                return
            
            prompt = create_prompt_template()
            parser = StrOutputParser()
            process_url = RunnableLambda(extract_data)
            
            # Create chain
            chain = process_url | prompt | model | parser
            
            # Process request
            with st.spinner("🔄 Processing your request..."):
                try:
                    start_time = time.time()
                    
                    result = chain.invoke({
                        "question": question_input,
                        "url": url_input
                    })
                    
                    end_time = time.time()
                    processing_time = round(end_time - start_time, 2)
                    
                    # Display results
                    st.success(f"✅ Answer generated in {processing_time} seconds!")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📝 Answer", "🔗 URL Info", "⚙️ Settings"])
                    
                    with tab1:
                        st.markdown("### 🤖 AI Response:")
                        st.markdown(f"**Question:** {question_input}")
                        st.markdown("**Answer:**")
                        st.write(result)
                        
                        # Copy button simulation
                        st.code(result, language=None)
                    
                    with tab2:
                        st.markdown("### 🔗 URL Information:")
                        parsed_url = urlparse(url_input)
                        st.write(f"**Domain:** {parsed_url.netloc}")
                        st.write(f"**Scheme:** {parsed_url.scheme}")
                        st.write(f"**Path:** {parsed_url.path}")
                        st.write(f"**Full URL:** {url_input}")
                    
                    with tab3:
                        st.markdown("### ⚙️ Request Settings:")
                        st.write(f"**Model:** llama3-8b-8192")
                        st.write(f"**Temperature:** {temperature}")
                        st.write(f"**Max Tokens:** {max_tokens}")
                        st.write(f"**Processing Time:** {processing_time}s")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.info("💡 Tips:\n- Check if the URL is accessible\n- Try a different webpage\n- Ensure your internet connection is stable")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit, LangChain, and GROQ</p>",
        unsafe_allow_html=True
    )

# Store chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Add chat history section
if st.session_state.chat_history:
    st.markdown("## 📚 Recent Queries")
    for i, (url, question, answer) in enumerate(st.session_state.chat_history[-3:]):
        with st.expander(f"Query {len(st.session_state.chat_history) - i}: {question[:50]}..."):
            st.write(f"**URL:** {url}")
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {answer[:200]}...")

if __name__ == "__main__":
    main()