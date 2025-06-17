from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate.from_template(
    "You are an helpful AI. Explain the topic {topic} like you are explaining to a child."
)

model = ChatGroq(model="llama3-8b-8192", temperature=0.9, max_tokens=100)

chat_model = template | model

result = chat_model.invoke({"topic": "Machine Learning"})

print(result.content)