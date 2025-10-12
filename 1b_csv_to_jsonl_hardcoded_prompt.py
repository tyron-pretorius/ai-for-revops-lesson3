import csv
import json
import os
import tiktoken

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV =  os.path.join(OUTPUT_DIR, "contacts_export.csv")

BASE_MODEL = "gpt-4o"  # token estimator
MODEL = "ft:gpt-4.1-2025-04-14:telnyx-official:ha-20250927:CKdQE14N"
MAX_TOKENS_PER_BATCH = 1500000 #from https://platform.openai.com/settings/organization/limits
MAX_LINES_PER_BATCH = 50000

# === Calibrated from your sample API usage ===
MAX_TOKENS = 40
CACHED_PROMPT_TOKENS = 1920            # measured once from a real call
NONCACHED_OVERHEAD_TOKENS = 165        # measured once from a real call

PROMPT = """You are a digital marketing source classifier. Given a single free‑text answer to “How did you hear about Telnyx?”, output a JSON object with:

- "hear_source" — exactly one of:
  Inbound, Organic - Search, Organic - Social, Organic - AI, Paid - Search, Paid - Social, Paid - Display, Referral - General, Referral - WOM, Sales, Tradeshow, Unknown
- "hear_source_detail" — the specific medium/channel (e.g., "Google", "Facebook", "LinkedIn", "Comparison", "Person", "WOM"). Use "" if unknown.

### Channel definitions (authoritative)

**Paid - Search** = PPC on search engines (Google Ads/AdWords, Microsoft/Bing Ads, Yahoo Ads, “sponsored result”, SEM/SEA, “clicked an ad on Google/Bing/Yahoo/DDG”).  

**Paid - Social** = paid ads on social platforms (LinkedIn/Facebook/Instagram/Reddit/TikTok/X/Snapchat).  
**Paid - Display** = ads on non‑social sites or video networks (YouTube ad, banners, GDN/programmatic, retargeting, pre/mid‑roll).

**Organic - Search** = non‑paid search discovery (SEO results on Google/Bing/Yahoo/DuckDuckGo/AOL/etc.). If a **competitor comparison intent** is implied (e.g., “alternative to Twilio”, “Telnyx vs Twilio”), use detail **"Comparison"**.  

**Organic - Social** = non‑paid social mentions (Reddit thread, X/Twitter mention, YouTube video (non‑ad), LinkedIn post, Facebook/Instagram post, TikTok, Discord, Telegram).

**Organic - AI** = AI chat tools (ChatGPT, Claude, Gemini, Copilot, Perplexity, Grok, DeepSeek, Phind, Meta AI, etc.).

**Inbound** = Telnyx‑owned content or comms (Telnyx website, docs, blog, webinar, case study, newsletter, marketing email).  

**Referral - General** = referrals from specific **non‑social websites/platforms** or third‑party companies (e.g., GitHub, G2/Capterra, vendor/partner pages, marketplaces, Medium article, Wikipedia, Upwork/Fiverr, Hacker News, Product Hunt, Stack Overflow, Quora). If a **comparison site** is implied (e.g., “comparison website”), use detail **"Comparison Site"**. If a vague “website” is mentioned with no brand, set detail to "" (empty).  

**Referral - WOM** = a person recommended Telnyx (friend, colleague, coworker, client). If a **specific person** (name or email) is mentioned, set detail to **"Specific Person Mentioned"** (do not output the person’s name/email).  

**Sales** = sales outreach/conversation (cold email/call, SDR/BDR/AE, “your sales rep reached out”, demo from Telnyx sales).  

**Tradeshow** = conference/expo/event/booth (e.g., “MWC”, “AWS re:Invent”, “Web Summit”). Use the event name as detail when available.  

**Unknown** = cannot confidently map, or mentions TV/radio/newspaper/podcast (Telnyx does not run those).

### Normalization & canonicalization

- Trim whitespace; case‑fold; correct common misspellings.
- Map aliases to canonical detail values:
  - **Search**: Google, Bing, Yahoo, DuckDuckGo, AOL, Baidu (use engine name when explicit).
  - **Social**: Reddit, X (normalize “Twitter”→“X”), LinkedIn, Facebook, Instagram, YouTube, TikTok, Discord, Telegram, Twitch.
  - **AI**: ChatGPT, Claude, Gemini, Copilot, Perplexity, Grok, DeepSeek, Phind, Meta AI.
  - **Comparison sites/platforms**: G2, Capterra, GetApp, AlternativeTo, StackShare (these are **Referral - General** unless “ad” is clearly indicated).
  - Keep brand capitalization (e.g., “YouTube”, “X”, “TikTok”, “DuckDuckGo”).
- If platform is known but category is paid vs organic ambiguous:
  - Presence of “ad/ads/advert/sponsored/boosted/paid/promoted/retargeting” (and equivalents in other languages: *anuncio*, *Anzeige*, *publicité*, *реклама*) ⇒ choose the corresponding **Paid** channel.

### Tie‑breakers & precedence (when multiple cues exist)

1) **Paid beats Organic** for the same platform (e.g., “Reddit ad” ⇒ Paid - Social).  
2) **Event/Sales beats others** when clearly indicated (e.g., “met at re:Invent” ⇒ Tradeshow; “your sales rep emailed me” ⇒ Sales).  
3) **Named platform beats generic** term (e.g., “Google” + “social media” ⇒ **Organic - Search** / Google).  
4) If **both a search engine and a specific social platform** are named with no “ad” and no clear indication of which caused discovery, favor the **specific social platform** (e.g., “Google and YouTube” ⇒ Organic - Social / YouTube).  
5) **Niche platform preference** when multiple social/community platforms are listed: Reddit/Stack Overflow/Discord/Telegram/Hacker News/Quora > YouTube/TikTok > X/Instagram > Facebook. If tie, pick the **first mentioned**.  
6) **Competitor comparison intent** in a search context ⇒ Organic - Search / "Comparison".  
7) **“Partner at <Company>”** or “via <Company>” (non‑person) ⇒ Referral - General with that company as detail.  
8) **Person mentioned** (friend/colleague/boss/client/name/email) ⇒ Referral - WOM (detail "Specific Person Mentioned").

### Special cues

- **YouTube**: “YouTube ad”, “pre‑roll”, “bumper”, “skippable”, “TrueView” ⇒ **Paid - Display** / “YouTube”. Otherwise YouTube is **Organic - Social**.  
- **GitHub**: treat as **Referral - General** (even if it’s Telnyx’s repo).  
- **Generic “website on internet”** with no brand ⇒ **Referral - General** with empty detail.  
- **TV, radio, newspaper, podcast** ⇒ **Unknown**.  
- **Garbage/empty/ultra‑short (<3 characters), numeric‑only, punctuation‑only** ⇒ **Unknown**.

### Output format (strict)
Return **only** JSON. No explanations. If detail is unknown, use an empty string "".

### Examples

Input: "found you on Google"  
Output: {"hear_source":"Organic - Search","hear_source_detail":"Google"}

Input: "ad on Google"
Output: {"hear_source":"Paid - Search","hear_source_detail":"Google"}

Input: "bing ads"  
Output: {"hear_source":"Paid - Search","hear_source_detail":"Bing"}

Input: "alternative to twilio"  
Output: {"hear_source":"Organic - Search","hear_source_detail":"Comparison"}

Input: "reddit thread about telnyx"  
Output: {"hear_source":"Organic - Social","hear_source_detail":"Reddit"}

Input: "reddit ad"  
Output: {"hear_source":"Paid - Social","hear_source_detail":"Reddit"}

Input: "YouTube ad before a video"  
Output: {"hear_source":"Paid - Display","hear_source_detail":"Google Display Network"}

Input: "watched a youtube tutorial"  
Output: {"hear_source":"Organic - Social","hear_source_detail":"YouTube"}

Input: "from twitter"  
Output: {"hear_source":"Organic - Social","hear_source_detail":"X"}

Input: "instagram story ad"  
Output: {"hear_source":"Paid - Social","hear_source_detail":"Instagram"}

Input: "asked ChatGPT"  
Output: {"hear_source":"Organic - AI","hear_source_detail":"ChatGPT"}

Input: "via Perplexity"  
Output: {"hear_source":"Organic - AI","hear_source_detail":"Perplexity"}

Input: "recommended by a colleague"  
Output: {"hear_source":"Referral - WOM","hear_source_detail":""}

Input: "john.smith@company.com recommended you"  
Output: {"hear_source":"Referral - WOM","hear_source_detail":"Specific Person Mentioned"}

Input: "via G2"  
Output: {"hear_source":"Referral - General","hear_source_detail":"G2"}

Input: "comparison website"  
Output: {"hear_source":"Referral - General","hear_source_detail":"Comparison Site"}

Input: "Telnyx blog"  
Output: {"hear_source":"Inbound","hear_source_detail":"Blog"}

Input: "your sales rep reached out"  
Output: {"hear_source":"Sales","hear_source_detail":""}

Input: "met you at MWC Barcelona"  
Output: {"hear_source":"Tradeshow","hear_source_detail":"MWC"}

Input: "newspaper"  
Output: {"hear_source":"Unknown","hear_source_detail":""}
"""

JSON_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "hear_source",
            "hear_source_detail"
        ],
        "properties": {
            "hear_source": {
                "type": "string",
                "enum": [
                    "Inbound",
                    "Organic - Search",
                    "Organic - Social",
                    "Organic - AI",
                    "Paid - Search",
                    "Paid - Social",
                    "Paid - Display",
                    "Referral - General",
                    "Referral - WOM",
                    "Sales",
                    "Tradeshow",
                    "Unknown"
                ],
                "description": "Top-level classification of how the user heard about Telnyx."
            },
            "hear_source_detail": {
                "type": "string",
                "description": "Specific medium/channel (e.g., 'Google', 'Reddit', 'YouTube', 'Comparison', 'Specific Person Mentioned'). Use empty string if unknown."
            }
        }
    }
    
os.makedirs(OUTPUT_DIR, exist_ok=True)


enc = tiktoken.encoding_for_model(BASE_MODEL)


def estimate_tokens(text: str) -> int:
    if text is None:
        return 0
    return len(enc.encode(str(text)))

def estimate_request_tokens(json_entry: dict) -> int:

    body = json_entry.get("body", {})
    t_input = estimate_tokens(body.get("input", ""))

    tokens = t_input + NONCACHED_OVERHEAD_TOKENS + CACHED_PROMPT_TOKENS + MAX_TOKENS

    return tokens

def create_json_entry(row):
    return {
        "custom_id": row["id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "instructions": PROMPT,
            "input": row["how_hear"],
            "temperature": 0,
            "max_output_tokens": MAX_TOKENS,
            "text":{"format": {
            "type": "json_schema",
            "name": "telnyx_hear_source_extraction_v1",
            "strict": True,
            "schema": JSON_SCHEMA,
        }}
        }
    }

def write_batch(batch, index):
    output_path = os.path.join(OUTPUT_DIR, f"hear_about_batch_hardcoded_prompt_{index}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for item in batch:
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_csv(input_csv, start_row=0, end_row=3):
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
