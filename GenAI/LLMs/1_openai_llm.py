from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo-instruct",
             temperature=0,
             max_tokens=50)

result = llm.invoke("Tell me a joke about cat.")

print(result)