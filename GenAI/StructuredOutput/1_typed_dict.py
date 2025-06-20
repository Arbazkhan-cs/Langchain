# import neccessary libraries
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# Initialize llm and define ChatModel
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    temperature=0.1,
)

model = ChatHuggingFace(llm=llm)

# Define Structured Class
class Info(TypedDict):
    name: Annotated[str, "Extract the name of the user."]
    email: Annotated[Optional[str], "Write down the user email address"]
    sentiment: Annotated[str, "Return sentiment of the feedback either positive, negative or neutral"]
    summary: Annotated[str, "A brief summary of the feedback"]

# Create structured output model
structured_output = model.with_structured_output(Info)

# Example
feedback_with_email = """
Hi, my name is Sarah Johnson and my email is sarah.johnson@gmail.com. 
I wanted to share my experience with your customer service team. 
They were absolutely fantastic! When I had an issue with my recent order, 
they resolved it within 24 hours and even provided a discount for the inconvenience. 
The representative was very polite and professional. I'm really impressed with the service quality.
"""

result1 = structured_output.invoke(feedback_with_email)
print("Feedback with Email:")
print(result1)
print("\n" + "="*50 + "\n")

# If we are using TypedDict then it only suggest the datatype to use but not validates it.
student_info = Info(name="arbazkhan", email=123, sentiment="pos", summary="blabla...") 
print(student_info)
# Here we can see that we have defined email as a string but still we can use integer value.