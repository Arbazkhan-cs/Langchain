from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

# load environment varaibel
load_dotenv()

# initialize llm
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  
    task="text-generation",
    temperature=0.5,
    max_tokens=100
)

# create chatmodel
model = ChatHuggingFace(llm=llm)

# Chat prompt template
template = [
    SystemMessage(content="You are a helpful AI assistant."),
    AIMessage(content="How can i assist you today?"),
    MessagesPlaceholder(variable_name="messages")
]

chat_prompt_template = ChatPromptTemplate(template)

# create chain
chat_chain = chat_prompt_template | model

#example
messages = [HumanMessage(content="Explain quantum computing in simple terms")]

result = chat_chain.invoke({"messages": messages})

print(result.content)