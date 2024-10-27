import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load data and set up embeddings
@st.cache_resource
def load_data():
    df = pd.read_csv("Test.csv", header=0)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [f"Q: {q}\nA: {a}" for q, a in zip(df['question'], df['answer'])]
    db = FAISS.from_texts(texts, embeddings)
    return db, texts, embeddings

db, texts, embeddings = load_data()

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Define the tools
@tool
def get_knowledge(question: str) -> str:
    """
    Provide answer by finding the answer from the database.
    If not found, returns a prompt to ask the user for an answer.
    """
    result = db.similarity_search(question)
    if result:
        return result[0].page_content
    else:
        return "I don't have an answer for this question yet. Could you provide one?"

@tool
def add_knowledge(question: str, answer: str) -> None:
    """
    Add the question and answer to the database if not provided.
    """
    global db, texts
    texts.append(f'Q: {question}\nA: {answer}')
    db = FAISS.from_texts(texts, embeddings)
    st.success("New knowledge added to the database.")

# Construct the prompt to include conversation history
def construct_prompt_with_history(input_text):
    conversation = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.conversation_history])
    return f"{conversation}\nQ: {input_text}"

# Setup the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI agent that answers questions based on a knowledge database.
       Follow these instructions carefully:
       1. First, use the get_knowledge tool to check if an answer exists in the database.
       2. If an answer is found, respond directly with it as follows:
          "Answer: <your answer here>"
       3. If no answer is found, respond with:
          "I don't have an answer for this question yet. Could you provide one?"
       4. Once the user provides an answer, use the add_knowledge tool to save it.
       Always respond with the final answer clearly, outside of tool-use tags.
    """),
    ("human", "{input}{agent_scratchpad}")
])


# Set up the model and agent
llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)
tools = [get_knowledge, add_knowledge]
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit app layout
st.title("Interactive Q&A AI Agent with Memory")
st.write("Ask a question, and if the AI can't find an answer, it will ask you to provide one. Previous conversations are remembered within this session.")

# User input
question = st.text_input("Enter your question:")

# Handle the response
if question:
    # Add conversation history to the prompt
    prompt_with_history = construct_prompt_with_history(question)
    
    # Query the agent
    response = agent_executor({"input": prompt_with_history})
    
    # Check if the agent is asking for new knowledge
    if "Could you provide one?" in response["output"]:
        st.write("I don't have an answer for this question yet.")
        user_answer = st.text_input("Please provide an answer:")
        
        # If user provides an answer, add it to the knowledge base and conversation history
        if user_answer:
            add_knowledge(question, user_answer)
            st.session_state.conversation_history.append((question, user_answer))
    else:
        # Display the answer and add the Q&A to conversation history
        st.write("Answer:", response["output"])
        st.session_state.conversation_history.append((question, response["output"]))

# Display conversation history
st.write("---")
st.write("Conversation History:")
for i, (q, a) in enumerate(st.session_state.conversation_history, start=1):
    st.write(f"{i}. Q: {q} - A: {a}")

st.write("---")
st.write("The database currently contains the following questions and answers:")

# Display current knowledge
for i, text in enumerate(texts, start=1):
    q, a = text.split("\n")
    st.write(f"{i}. {q.replace('Q: ', '')} - {a.replace('A: ', '')}")
