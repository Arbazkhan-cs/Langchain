from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o",
                    temperature=0.5,
                    max_tokens=50)

result = model.invoke("Tell me a joke about cat.")

print(result.content)