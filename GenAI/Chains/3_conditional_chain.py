# import neccessary libraires
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# load model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# load parsers
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)
str_parser = StrOutputParser()

# Crete PromptTemplates
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# create classifier chain
classifier_chain = prompt1 | model | pydantic_parser

# creta conditional chain
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | str_parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

# Example 1
result = chain.invoke({'feedback': 'This is a beautiful phone'})
print(result)
print("="*50)

# Exmample 2
result = chain.invoke({'feedback': 'This is a terrible phone'})
print(result)
print("="*50)

chain.get_graph().print_ascii()