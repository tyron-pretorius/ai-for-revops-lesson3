import csv
import json
import os
import tiktoken

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV =  os.path.join(OUTPUT_DIR, "Input.csv")

BASE_MODEL = "gpt-4o"  # token estimator
MODEL = "gpt-5"
MAX_TOKENS_PER_BATCH = 1500000 #from https://platform.openai.com/settings/organization/limits
MAX_LINES_PER_BATCH = 50000
PROMPT_ID = "pmpt_68df1b8d8d2c819381cee34b826632170d9dc7e1b8052f56"

# === Calibrated from your sample API usage ===
MAX_TOKENS = 100
PROMPT_TOKENS = 1940            # measured once from a real call
OVERHEAD_TOKENS = 145        # measured once from a real call

os.makedirs(OUTPUT_DIR, exist_ok=True)


enc = tiktoken.encoding_for_model(BASE_MODEL)


def estimate_tokens(text: str) -> int:
    if text is None:
        return 0
    return len(enc.encode(str(text)))

def estimate_request_tokens(json_entry: dict) -> int:
    body = json_entry.get("body", {})
    t_input = estimate_tokens(body.get("input", ""))

    tokens = t_input + OVERHEAD_TOKENS + PROMPT_TOKENS + MAX_TOKENS

    return tokens

def create_json_entry(row):
    return {
        "custom_id": row["id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "prompt": {"id": PROMPT_ID},
            "input": row["how_hear"],
            "max_output_tokens": MAX_TOKENS,
        }
    }

def write_batch(batch, index):
    output_path = os.path.join(OUTPUT_DIR, f"hear_about_batch_api_{index}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in batch:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_csv(input_csv, start_row=0, end_row=None):
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        all_rows = list(reader)
        selected_rows = all_rows[start_row:end_row]

        batch = []
        batch_token_count = 0
        batch_index = 0

        for row in selected_rows:
            json_entry = create_json_entry(row)

            # Estimate tokens for this request
            req_tokens = estimate_request_tokens(json_entry)

            # If adding this item would exceed token/line limits, flush the current batch first
            if batch and (
                batch_token_count + req_tokens > MAX_TOKENS_PER_BATCH
                or len(batch) >= MAX_LINES_PER_BATCH
            ):
                write_batch(batch, batch_index)
                batch_index += 1
                batch = []
                batch_token_count = 0

            # Edge case: single request larger than budget; write it alone to avoid blocking
            if not batch and req_tokens > MAX_TOKENS_PER_BATCH:
                write_batch([json_entry], batch_index)
                batch_index += 1
                continue

            batch.append(json_entry)
            batch_token_count += req_tokens

        if batch:
            write_batch(batch, batch_index)

if __name__ == "__main__":
    process_csv(INPUT_CSV)
