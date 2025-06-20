from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Define OutputParser
parser = StrOutputParser()

# Initialize promptTemplate
template1 = PromptTemplate(
    template="You are a helpfull AI Researcher, Provide a deep research about the given topic {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="You are a helpfull AI notes maker, Provide me the 5 important notes about the given text {text}",
    input_variables=["text"]
) 

# Create Sequential Chain
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "Attention is all you need!"})
print(result)
print(chain.get_graph().print_ascii())
