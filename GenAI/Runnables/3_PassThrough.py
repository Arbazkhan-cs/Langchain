# Import neccessary libraires
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
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
    template="create a extream funny joke on the topic {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Explain the joke\n<{joke}>",
    input_variables=["joke"]
)

# create chain
joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explaination": RunnableSequence(prompt2, model, parser)
})
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "cat"})
print(result)
print("Joke =", result["joke"])
print("\nExplaination =", result["explaination"])
