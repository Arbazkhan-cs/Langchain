import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from code_editor import code_editor
import tempfile
import os
from typing import Dict, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS with better organization and mobile responsiveness
st.markdown("""
<style>
    :root {
        --primary-color: #2ecc71;
        --secondary-color: #27ae60;
        --background-dark: #1a1a1a;
        --code-background: #2d2d2d;
        --tab-background: #3d3d3d;
    }
    
    .main {
        background-color: var(--background-dark);
        color: #ffffff;
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .stCodeBlock {
        background-color: var(--code-background) !important;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 20px;
        transition: all 0.3s;
        padding: 0.5rem 1rem;
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .stButton button:hover {
        background-color: var(--secondary-color) !important;
        transform: scale(1.02);
    }
    
    .stFileUploader label {
        color: var(--primary-color) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--tab-background) !important;
        color: white !important;
        border-radius: 8px !important;
        transition: all 0.3s !important;
        min-width: 120px;
        text-align: center;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
    }
    
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            min-width: 100px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Type definitions for better code structure
MessageType = Dict[str, Union[str, List[str]]]
SessionState = Dict[str, Union[str, List[MessageType]]]

def init_session_state() -> None:
    """Initialize session state with default values."""
    if "message_log" not in st.session_state:
        st.session_state.message_log = [{
            "role": "ai", 
            "content": "Hi! I'm DeepSeek Pro. Ready to tackle complex coding challenges? 💻",
            "code_blocks": []
        }]
    
    if "code_context" not in st.session_state:
        st.session_state.code_context = ""

def handle_file_upload(uploaded_file) -> None:
    """Handle file upload with error handling."""
    if uploaded_file:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                st.session_state.code_context = open(temp_file.name).read()
            os.unlink(temp_file.name)
        except Exception as e:
            logger.error(f"Error handling file upload: {str(e)}")
            st.error("Failed to process uploaded file. Please try again.")

def create_code_editor() -> Dict:
    """Create and configure the code editor component."""
    try:
        return code_editor(
            st.session_state.code_context,
            lang="python",
            height=400,
            theme="dark",
            key="code_editor",
            buttons=[
                {
                    "name": "Send to Chat",
                    "feather": "Send",
                    "primary": True,
                    "hasText": True,
                    "key": "send_to_chat"
                }
            ]
        )
    except Exception as e:
        logger.error(f"Error creating code editor: {str(e)}")
        st.error("Failed to initialize code editor. Please refresh the page.")
        return {}

def handle_editor_button_click(editor_content: Dict) -> None:
    """Handle code editor button clicks."""
    if editor_content and 'type' in editor_content:
        if editor_content['type'] == 'button' and editor_content.get('key') == 'send_to_chat':
            st.session_state.message_log.append({
                "role": "user",
                "content": f"Analyze this code:\n{editor_content['text']}",
                "code_blocks": []
            })
            st.rerun()

def get_system_prompt() -> SystemMessagePromptTemplate:
    """Generate system prompt based on user settings."""
    prompt = (
        "You are an expert AI coding assistant with advanced capabilities. "
        "Follow these rules:\n"
        "1. Provide clean, efficient solutions with error handling\n"
        "2. Include strategic debug points and logging\n"
        "3. Optimize for performance and readability\n"
        "4. Support multiple programming languages\n"
        f"5. {'Explain concepts step-by-step' if st.session_state.explain_toggle else 'Be concise'}\n"
        f"6. {'Auto-detect potential errors' if st.session_state.debug_toggle else ''}\n"
        f"7. {'Suggest optimizations' if st.session_state.optimize_toggle else ''}\n"
        "Always format code with syntax highlighting and markdown."
    )
    return SystemMessagePromptTemplate.from_template(prompt)

@st.cache_resource
def get_llm_engine(model: str, temperature: float, max_tokens: int) -> ChatOllama:
    """Initialize and cache the LLM engine."""
    try:
        return ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"Error initializing LLM engine: {str(e)}")
        st.error("Failed to initialize AI model. Please check if Ollama is running.")
        return None

def display_message(message: MessageType) -> None:
    """Display chat messages with syntax highlighting and error handling."""
    try:
        with st.chat_message(message["role"]):
            content = message["content"]
            if '```' in content:
                parts = content.split('```')
                for i, part in enumerate(parts):
                    if i % 2 == 1:
                        lang, code = part.split('\n', 1) if '\n' in part else ('', part)
                        st.code(code, language=lang.strip() or 'python')
                    else:
                        st.markdown(part)
            else:
                st.markdown(content)
    except Exception as e:
        logger.error(f"Error displaying message: {str(e)}")
        st.error("Failed to display message. Please try again.")

def main():
    """Main application logic with error handling and state management."""
    try:
        # Initialize session state
        init_session_state()
        
        # Set up page title and description
        st.title("🧠 DeepSeek Code Companion Pro")
        st.caption("🚀 Your Ultimate AI Pair Programmer with Advanced Debugging & Code Analysis")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            selected_model = st.selectbox(
                "Choose Model",
                ["deepseek-r1:1.5b", "deepseek-r1:7b"],
                index=0
            )
            
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
            max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100)
            
            st.checkbox("Explain Step-by-Step", True, key="explain_toggle")
            st.checkbox("Auto-Debug Mode", True, key="debug_toggle")
            st.checkbox("Code Optimization", True, key="optimize_toggle")
            
            st.divider()
            
            uploaded_file = st.file_uploader("Upload Code File", type=["py", "js", "java", "cpp"])
            handle_file_upload(uploaded_file)
            
            st.divider()
            st.markdown("### Hotkeys")
            st.markdown("""
            - `Ctrl+Enter`: Execute code
            - `Ctrl+S`: Save session
            - `Ctrl+D`: Toggle dark mode
            """)
            
            st.divider()
            
            # Additional features
            if st.button("Clear Chat History"):
                st.session_state.message_log = [{
                    "role": "ai", 
                    "content": "Chat history cleared. Ready for new tasks!",
                    "code_blocks": []
                }]
                st.rerun()
            
            if st.button("Export Session"):
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.message_log])
                st.download_button(
                    label="Download Chat",
                    data=chat_history,
                    file_name="deepseek_chat_history.txt",
                    mime="text/plain"
                )
        
        # Initialize LLM engine
        llm_engine = get_llm_engine(selected_model, temperature, max_tokens)
        if not llm_engine:
            return
        
        # Create tabs
        tab1, tab2 = st.tabs(["💬 Chat", "📝 Code Editor"])
        
        # Code editor tab
        with tab2:
            editor_content = create_code_editor()
            handle_editor_button_click(editor_content)
            st.session_state.code_context = editor_content.get('text', '')
        
        # Chat interface
        with tab1:
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.message_log:
                    display_message(message)
            
            user_query = st.chat_input("Type your coding question here...")
            
            if user_query:
                full_query = f"{st.session_state.code_context}\n\n{user_query}" if st.session_state.code_context else user_query
                
                st.session_state.message_log.append({
                    "role": "user",
                    "content": full_query,
                    "code_blocks": []
                })
                
                with st.spinner("🧠 Processing..."):
                    try:
                        prompt_chain = ChatPromptTemplate.from_messages([
                            get_system_prompt(),
                            *[
                                HumanMessagePromptTemplate.from_template(msg["content"])
                                if msg["role"] == "user"
                                else AIMessagePromptTemplate.from_template(msg["content"])
                                for msg in st.session_state.message_log[:-1]
                            ]
                        ])
                        
                        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
                        ai_response = processing_pipeline.invoke({})
                        
                        # Error detection
                        error_keywords = ["error", "exception", "warning", "traceback"]
                        if any(keyword in ai_response.lower() for keyword in error_keywords):
                            ai_response = f"🚨 Potential Issue Detected!\n\n{ai_response}"
                        
                        st.session_state.message_log.append({
                            "role": "ai",
                            "content": ai_response,
                            "code_blocks": []
                        })
                        
                        st.rerun()
                    
                    except Exception as e:
                        logger.error(f"Error generating response: {str(e)}")
                        st.error("Failed to generate response. Please try again.")
        
        st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()