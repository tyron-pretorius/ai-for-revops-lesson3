from openai import OpenAI
import os, json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

batches = client.batches.list(limit=10)
print(json.dumps(batches.model_dump(), indent=2))
