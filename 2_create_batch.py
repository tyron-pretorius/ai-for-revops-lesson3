from openai import OpenAI
import os, json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "hear_about_batch_api_0.jsonl")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
batch_input_file = client.files.create(
    file=open(file_path, "rb"),
    purpose="batch"
)

print("Batch Input File:")
print(json.dumps(batch_input_file.model_dump(), indent=2))

batch_input_file_id = batch_input_file.id
response = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/responses",
    completion_window="24h",
    metadata={
        "description": "3 values with prompt id and gpt5 100 max tokens"
    }
)

print("\nBatch Response:")
print(json.dumps(response.model_dump(), indent=2))

# Extract and print the batch ID
batch_id = response.id
print(f"\nBatch ID: {batch_id}")

# Save batch ID to file for use by other scripts
batch_id_file = os.path.join(current_dir, "latest_batch_id.txt")
with open(batch_id_file, "w") as f:
    f.write(batch_id)

print(f"Batch ID saved to: {batch_id_file}")