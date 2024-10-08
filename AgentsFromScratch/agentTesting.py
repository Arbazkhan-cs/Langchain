from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import base64, requests, subprocess, tempfile, re, importlib, sys
from PIL import Image
from io import BytesIO
import os

class ProAgent:
    def __init__(self) -> None:
        self.llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, max_tokens=4000)
        self.template = [("system", "You are a python code generator. Respond only with executable Python code, no explanations or comments except for required pip installations at the top"), ("user", "Generate python code to {prompt}. If you need to use any external libraires, include a comment at the top of the code listing required pip installations.")]

        self.promptTemplate = ChatPromptTemplate.from_messages(self.template)
        self.model = self.promptTemplate | self.llm
        

    def generate_code(self, prompt):
        result = self.model.invoke({"prompt": prompt})
        code = re.sub(r'^```python\n|```$', '', result.content, flags=re.MULTILINE)
        code_lines = code.split("\n")
        while code_lines and not (code_lines[0].startswith('import') or code_lines[0].startswith('from') or code_lines[0].startswith("#")):
            code_lines.pop(0)
        
        return '\n'.join(code_lines)

    def install_libraries(self, code):
        libraries = re.findall(r'#\s*pip install\s+([\w-]+)', code)
        if libraries:
            print("Installing required libraries...")
            for lib in libraries:
                try:
                    importlib.import_module(lib.replace('-', '_'))
                    print(f"{lib} is already installed.")
                except ImportError:
                    print(f"Installing {lib}...")
                    # print(sys.executable)
                    subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            
            print("Libraries Installed Successfully")
    
    def execute_code(self, code):
        with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True, timeout=30)
            output = result.stdout
            err = result.stderr
        except subprocess.TimeoutExpired:
            output = ""
            err = "Execution timed out after 30 seconds."
        except Exception as e:
            output = ""
            err = f"An error occurred during execution: {str(e)}"
        finally:
            os.unlink(temp_file_path)

        return output, err

    def run(self, prompt):
        print(f"Generating Code For: {prompt}")
        code = self.generate_code(prompt=prompt)
        print("Generate code:")
        print(code)
        print("Executing code...")
        self.install_libraries(code)
        output, err = self.execute_code(code)

        if output:
            print("Output:")
            print(output)
        if err:
            print("Error:")
            print(err)


if __name__ == "__main__":
    agent = ProAgent()
    agent.run("""make a detailed deck on the best forms of leadership with atleast 10 slides and save it to a pptx called leadership.pptx""")
