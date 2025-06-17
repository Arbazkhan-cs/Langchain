from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

template = [("system", "You are an helpfull AI. Your name is {name}, {name} explains topics in simple way."),
            ("human", "Hi, how are you?"),
            ("ai", "hello, i am good. How can i assist you today?"),
            ("human", "{msg}")]

chat_prompt_template = ChatPromptTemplate(template)

model = ChatGroq(model="llama3-8b-8192", temperature=0.9, max_tokens=100)

chat_model = chat_prompt_template | model

result = chat_model.invoke({
    "name": "Bob",
    "msg": "What is your name, explain me Machine Leaning"
})

print(result.content)