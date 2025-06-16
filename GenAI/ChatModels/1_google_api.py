from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", 
                               temperature=1, 
                               max_tokens=50)

result = model.invoke("Tell me a joke about cats.")

print(result.content)