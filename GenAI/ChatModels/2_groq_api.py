from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", 
                temperature=0.7, 
                max_tokens=50)

result = model.invoke("Tell me a joke about cat.")

print(result.content)