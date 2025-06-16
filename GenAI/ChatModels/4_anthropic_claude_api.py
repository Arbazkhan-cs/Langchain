from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-opus-20240229",
                        temperature=0.7,
                        max_tokens=50)

result = model.invoke("Tell me a joke about cat.")

print(result.content)