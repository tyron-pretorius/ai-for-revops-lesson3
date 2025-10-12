import pandas as pd
import json
from openai import OpenAI
import ast
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read output file ID from file
current_dir = os.path.dirname(os.path.abspath(__file__))
output_file_id_file = os.path.join(current_dir, "latest_output_file_id.txt")

with open(output_file_id_file, "r") as f:
    output_file_id = f.read().strip()

print(f"Using output file ID: {output_file_id}")
output_file = client.files.content(output_file_id)

# Save raw output for reference
output_jsonl_path = os.path.join(current_dir, "output_hear_about.jsonl")
with open(output_jsonl_path, "w") as f:
    f.write(output_file.text)
print(f"Raw output saved to: {output_jsonl_path}")

# Parse the response data
lines = output_file.text.strip().split("\n")
data = [json.loads(line) for line in lines]

# Process each record from OpenAI response
records = []
for item in data:
    custom_id = item.get("custom_id")
    response_body = item["response"]["body"]
    
    # Find the message output (skip reasoning outputs)
    message_output = None
    for output_item in response_body["output"]:
        if output_item.get("type") == "message" and "content" in output_item:
            message_output = output_item
            break
    
    content = message_output["content"][0]["text"]
    parsed = ast.literal_eval(content)
    hear_source = parsed.get('hear_source', '')
    hear_source_detail = parsed.get('hear_source_detail', '')
    
    # Extract token usage if available
    usage = response_body.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    
    # Calculate costs (matching poll_batch.py pricing)
    input_cost_per_million = 0.625  # $0.625 per million input tokens
    output_cost_per_million = 5.0  # $5 per million output tokens
    cost_prompt = (prompt_tokens / 1_000_000) * input_cost_per_million
    cost_completion = (completion_tokens / 1_000_000) * output_cost_per_million
    total_cost = cost_prompt + cost_completion

    records.append({
        "custom_id": custom_id,
        "hear_source": hear_source,
        "hear_source_detail": hear_source_detail,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(total_cost, 6)
    })

# Create DataFrame from OpenAI results
df_new = pd.DataFrame(records)

# Load existing CSV
input_path = os.path.join(current_dir,"Input.csv")
output_path = os.path.join(current_dir,"output_hear_about.csv")
if os.path.exists(input_path):
    df_existing = pd.read_csv(input_path)

    # Perform INNER JOIN: only keep rows where id is found in new data
    df_merged = df_existing.merge(df_new, how='inner', left_on="id", right_on="custom_id")

    # Drop redundant custom_id column if desired
    df_merged.drop(columns=["custom_id"], inplace=True)

    # Save to output path
    df_merged.to_csv(output_path, index=False)
    print(f"Filtered and merged results saved to {output_path}")
else:
    print(f"File not found at {input_path}")