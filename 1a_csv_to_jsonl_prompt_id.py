import csv
import json
import os
import tiktoken

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV =  os.path.join(OUTPUT_DIR, "contacts_export.csv")

BASE_MODEL = "gpt-4o"  # token estimator
MODEL = "gpt-4o"
MAX_TOKENS_PER_BATCH = 15000000000 #from https://platform.openai.com/settings/organization/limits
MAX_LINES_PER_BATCH = 50000
PROMPT_ID = "pmpt_68d708122c0c81979c4ad6ad41ebc4ec0351f203636fba6f"
MAX_TOKENS = 40

# === Calibrated from your sample API usage ===
# usage.input_tokens = 2087; usage.input_tokens_details.cached_tokens = 1920
# -> Non-cached overhead ≈ 2087 - 1920 - tokens(input_text) ~= 167 when input empty.
CACHED_PROMPT_TOKENS = 1920            # measured once from a real call
NONCACHED_OVERHEAD_TOKENS = 165        # measured once from a real call with empty input

# If True: add the cached prompt tokens only ONCE per batch (recommended).
# If False: add the cached prompt tokens to EVERY request (very conservative).
ASSUME_PROMPT_CACHED = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


enc = tiktoken.encoding_for_model(BASE_MODEL)


def estimate_tokens(text: str) -> int:
    if text is None:
        return 0
    return len(enc.encode(str(text)))

def estimate_request_tokens(json_entry: dict, is_first_in_batch: bool) -> int:
    """
    Estimate request tokens for batching:
      tokens ~= tokens(input) + max_output + NONCACHED_OVERHEAD_TOKENS
      + (CACHED_PROMPT_TOKENS once per batch if ASSUME_PROMPT_CACHED else each request)
    """
    body = json_entry.get("body", {})
    t_input = estimate_tokens(body.get("input", ""))

    tokens = t_input + MAX_TOKENS + NONCACHED_OVERHEAD_TOKENS

    if ASSUME_PROMPT_CACHED:
        if is_first_in_batch:
            tokens += CACHED_PROMPT_TOKENS
    else:
        tokens += CACHED_PROMPT_TOKENS

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
            "temperature": 0,
            "max_output_tokens": MAX_TOKENS,
        }
    }

def write_batch(batch, index):
    output_path = os.path.join(OUTPUT_DIR, f"hear_about_batch_api_{index}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in batch:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_csv(input_csv, start_row=0, end_row=1000):
    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        all_rows = list(reader)
        selected_rows = all_rows[start_row:end_row]

        batch = []
        batch_token_count = 0
        batch_index = 0

        for row in selected_rows:
            json_entry = create_json_entry(row)

            # Predict tokens if we were to add this as the *next* item in the batch
            is_first_in_batch = (len(batch) == 0)
            req_tokens = estimate_request_tokens(json_entry, is_first_in_batch=is_first_in_batch)

            # If adding this item would exceed token/line limits, flush the current batch first
            if batch and (
                batch_token_count + req_tokens > MAX_TOKENS_PER_BATCH
                or len(batch) >= MAX_LINES_PER_BATCH
            ):
                write_batch(batch, batch_index)
                batch_index += 1
                batch = []
                batch_token_count = 0
                # Recompute req_tokens for the new batch's first-line case
                req_tokens = estimate_request_tokens(json_entry, is_first_in_batch=True)

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
