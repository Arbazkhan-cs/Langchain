# Import neccessary libraires
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# load model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# define parser
parser = StrOutputParser()

# Create PromptTemplate
prompt1 = PromptTemplate(
    template="create linkedin post for the topic {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="create tweeter post for the topic {topic}",
    input_variables=["topic"]
)

# create chain
chain = RunnableParallel({
    "linkedin": RunnableSequence(prompt1, model, parser),
    "tweeter": RunnableSequence(prompt2, model, parser)
})
result = chain.invoke({"topic": "AI"})
print(result)
print("Linkedin Post:", result["linkedin"])
print("\nTweeter Post:", result["tweeter"])
