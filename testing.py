import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Read API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create client
client = genai.Client(api_key=api_key)

# Send prompt
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain AST parsing in simple terms."
)

# Print response
print(response.text)