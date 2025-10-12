from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read batch ID from file
current_dir = os.path.dirname(os.path.abspath(__file__))
batch_id_file = os.path.join(current_dir, "latest_batch_id.txt")

with open(batch_id_file, "r") as f:
    batch_id = f.read().strip()

print(f"Cancelling batch: {batch_id}")

# Cancel the batch
cancelled_batch = client.batches.cancel(batch_id)

print(f"Batch cancellation status: {cancelled_batch.status}")
print(f"Cancelling at: {cancelled_batch.cancelling_at}")

if cancelled_batch.cancelled_at:
    print(f"Cancelled at: {cancelled_batch.cancelled_at}")
    print("✅ Batch successfully cancelled!")
else:
    print("⏳ Batch cancellation is in progress...")
