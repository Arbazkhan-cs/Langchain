from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceHub(repo_id="HPAI-BSC/Qwen2.5-Aloe-Beta-72B", model_kwargs={"temperature":0.7, "max_length":500})

output = llm("What is ML?")
print(output)