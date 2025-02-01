from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
import os
from typing import List, Dict, Any
from http import HTTPStatus

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate environment variables
required_env_vars = ['GROQ_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize LLM
try:
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.5,
        api_key=os.getenv('GROQ_API_KEY')
    )
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Prompt Template
SYSTEM_PROMPT = """
Task: Provide deep syllabus for the given subject by following below guidelines:
    -> Generate deep syllabus for the given subject.
    -> The syllabus should be in the form of json which consist of list of topics as below:
 <startJson>
    "subject": "the subject which is given by the user",
    "syllabus": ["topic1", "topic2", "topic3"],
 <endJson>
Example:
 <startJson>
    "subject": "Software Engineering",
    "syllabus": ["Introduction to software engineering", "Software crises", "Software Life Cycle Model", "Waterfall Model", "Prototype Model", "Spiral Model", "Agile Model", "Software Requirement Analysis and Specification", ...],
 </endJson>
Output: The result should a JSON object with the information as specified, without any additional information.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Subject: {subject}")
])

# Create chain
model = prompt_template | llm

def validate_request(data: List[Dict[str, Any]]) -> List[str]:
    """Validate the incoming request data."""
    errors = []
    if not isinstance(data, list):
        errors.append("Request body must be a list of objects")
        return errors
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item at index {idx} must be an object")
        elif "subject" not in item:
            errors.append(f"Missing 'subject' field in item at index {idx}")
        elif not isinstance(item.get("subject"), str):
            errors.append(f"'subject' must be a string in item at index {idx}")
        elif not item.get("subject").strip():
            errors.append(f"'subject' cannot be empty in item at index {idx}")
    
    return errors

def process_item(subject: str) -> Dict[str, Any]:
    """Process a single subject and generate its syllabus."""
    try:
        logger.info(f"Processing subject: {subject}")
        response = model.invoke({"subject": subject})
        output = dict(response).get('output', '')
        
        # Log the raw output for debugging
        logger.debug(f"Raw output from LLM for subject '{subject}': {output}")
        
        return output
    except Exception as e:
        logger.error(f"Error processing subject '{subject}': {str(e)}")
        return {
            "error": "An unexpected error occurred during processing",
            "details": str(e)
        }

@app.route('/SwaySyllabusGenerator', methods=['POST'])
def generate_syllabus():
    """Generate syllabi for multiple subjects."""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON in request body"}), HTTPStatus.BAD_REQUEST
        
        # Validate request
        validation_errors = validate_request(data)
        if validation_errors:
            return jsonify({"errors": validation_errors}), HTTPStatus.BAD_REQUEST
        
        # Process each subject
        responses = []
        for item in data:
            response = process_item(subject=item["subject"])
            responses.append(response)
        
        return jsonify(responses), HTTPStatus.OK

    except Exception as e:
        logger.error(f"Error in generate_syllabus endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), HTTPStatus.INTERNAL_SERVER_ERROR

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        "message": "Welcome to Sway Syllabus Generator API",
        "version": "1.0",
        "endpoints": {
            "/SwaySyllabusGenerator": {
                "method": "POST",
                "description": "Generate syllabus for multiple subjects"
            }
        }
    })

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "details": str(error)
    }), HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)