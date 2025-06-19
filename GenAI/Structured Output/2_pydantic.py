from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

# load environment variable
load_dotenv()

# Define Model
model = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# Define pydantic structure
class Info(BaseModel):
    name: str = Field(description="Name of the person")
    email: Optional[str] = Field(description="Email of the person")
    phone_no: Optional[str] = Field(description="Phone number of the person")  # Changed to str
    feedback_summary: str = Field(description="Summary of the feedback provided by the person")

# define structured output model
structured_output_model = model.with_structured_output(Info)

# Example 1: Your current example (with name added)
query1 = """
Hi, my name is Arbaz Khan. I've been meaning to share my experience with this particular product ever since I received it. To begin with, I wasn't sure about how well it would perform, but the first few days of use changed my perspective. Initially, I thought it might not live up to the description on the website, but it's been surprisingly durable. There was a slight hiccup with delivery—my order showed up a bit later than expected, but the packaging was intact, and everything inside was in perfect condition.

A few days into using the product, I realized how practical it actually is, and it's become a part of my daily routine. However, there's always room for improvement, right? While the design is sleek and modern, I feel like it could use a bit more versatility, perhaps in terms of color options. arbaz123@email.com I did consider reaching out to customer support to inquire about availability, but I didn't have to wait long. 91-786543-7890 I found all the answers I needed in the FAQ section, which was really thorough and helpful.

Overall, I'm quite happy with the experience and would recommend it to others. The product performs well despite minor delivery delays, and customer support resources are helpful.
"""

print("Example 1 - Complete Information:")
response1 = structured_output_model.invoke(query1)
print(response1)
print(f"Type: {type(response1)}")
print("\n" + "="*60 + "\n")

# Example 2: Feedback with only email
query2 = """
Hello, I'm Sarah Johnson. I recently purchased your premium headphones and wanted to share my thoughts. 
The sound quality is absolutely amazing - the bass is deep and the highs are crystal clear. 
I use them daily for both work calls and music, and they've exceeded my expectations. 
The battery life is impressive too, lasting the entire workday without needing a charge.
My email is sarah.j@techcorp.com if you need to follow up on this feedback.
The only minor issue is that they're a bit tight initially, but they've loosened up with use.
"""

print("Example 2 - With Email Only:")
response2 = structured_output_model.invoke(query2)
print(response2)
print(f"Type: {type(response2)}")
print("\n" + "="*60 + "\n")

# # It will through a validation error because it also validate the data type of the variables, here we have used integer value but email is expacting string value.
# obj1 = Info(name="Arbaz khan", email=123, phone_no="876XXX321", feedback_summary="blabla...") 
# print(obj1)

# Now it will work as expected
obj2 = Info(name="Arbaz khan", email="arbazkhan@yahoo.com", phone_no="8700XXXX42", feedback_summary="blabla...")
print(obj2)
print("Seperate Email:", obj2.email)
print(type(obj2))
