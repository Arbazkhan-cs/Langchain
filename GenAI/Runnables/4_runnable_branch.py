# Import neccessary libraires
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
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
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the feedback either positive, negative, or neutral")
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)
str_parser = StrOutputParser()

# Create PromptTemplate
prompt1 = PromptTemplate(
    template="provide the sentiment of the given feedback \n {feedback}, \n {format_instruction}",
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

# create chain
classifier_chain = RunnableSequence(prompt1, model, pydantic_parser)
conditional_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", RunnableSequence(prompt2, model, str_parser)),
    (lambda x: x.sentiment == "negative", RunnableSequence(prompt3, model, str_parser)),
    RunnableLambda(lambda x: "can not identify the sentiment!")
)
final_chain = RunnableSequence(classifier_chain, conditional_chain)

result = final_chain.invoke({"feedback": "the phone is realy good"})
print(result)
