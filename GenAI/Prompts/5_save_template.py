from langchain_core.prompts import PromptTemplate

# Define PromptTemplate
template = """Please summarize the research paper titled "{paper_title}" with the following specifications:
Explanation Style: {input_style}
Explanation Length: {input_length}
1. Mathematical Details:
    - Include relevent mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
    - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with "Insufficent information available" instead of guessing.
Ensure the summary clear, accurate, and aligned with the provided style and length.
"""
prompt_template = PromptTemplate.from_template(template)

# save prompt template
prompt_template.save("research_paper_prompt_template.json")