from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read batch ID from file
current_dir = os.path.dirname(os.path.abspath(__file__))
batch_id_file = os.path.join(current_dir, "latest_batch_id.txt")

#https://platform.openai.com/docs/pricing?latest-pricing=batch
input_cost_per_million = 0.625  # $0.625 per million input tokens
output_cost_per_million = 5.0  # $5 per million output tokens


with open(batch_id_file, "r") as f:
    batch_id = f.read().strip()

# batch_id = "batch_68e99fcfcdf081908897d26c83734b8d"

print(f"Polling batch: {batch_id}")
batch = client.batches.retrieve(batch_id)
print(json.dumps(batch.model_dump(), indent=2))

# Calculate and display batch cost when completed
if batch.status == "completed" and batch.usage:

    output_file_id = batch.output_file_id
    output_file_id_file = os.path.join(current_dir, "latest_output_file_id.txt")
    with open(output_file_id_file, "w") as f:
        f.write(output_file_id)
    print(f"\nOutput file ID: {output_file_id}")
    print(f"Output file ID saved to: {output_file_id_file}")
    
    input_tokens = batch.usage["input_tokens"]
    output_tokens = batch.usage["output_tokens"]
    total_tokens = batch.usage["total_tokens"]
    
    # Calculate costs
    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost
    
    print(f"\n=== BATCH COST SUMMARY ===")
    print(f"Input tokens: {input_tokens:,}")
    print(f"Output tokens: {output_tokens:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Input cost: ${input_cost:.6f}")
    print(f"Output cost: ${output_cost:.6f}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Request counts - Completed: {batch.request_counts.completed}, Failed: {batch.request_counts.failed}, Total: {batch.request_counts.total}")