from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Initialize promptTemplate
template1 = PromptTemplate(
    template="You are a helpfull AI Researcher, Provide a deep research about the given topic {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="You are a helpfull AI notes maker, Provide me the 5 notes about the givene text {text}",
    input_variables=["text"]
)

# Without StrOutputParser: We have to use rsult.content to send the output to the next prompt
input_topic = "Attention is all you need"
prompt1 = template1.invoke({"topic": input_topic})
result_1 = model.invoke(prompt1)
prompt2 = template2.invoke({"text": result_1.content})
result_2 = model.invoke(prompt2)

print(result_2.content)
print("="*10)
# Here we can see that if we are working on a more complex project then, it will become difficult to manage and create a big chain. But with the helfull OutputParsers we can do the same task in a single line using pipelines. 

# With StrOutputParser
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": input_topic})
print(result)
# By StrOutputParser we can directly send the string output to the next prompt
