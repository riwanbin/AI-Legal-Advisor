"""
AI Legal Advisor - Lab Script 01: Verify Gemini API Connection
================================================================
This script validates that:
1. The Gemini API key in .env is working.
2. We can call gemini-2.5-flash-lite for text generation.
3. We can call gemini-embedding-001 for text embeddings.

Run: .venv\\Scripts\\python lab\\01_verify_gemini.py
"""

import os
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# -- Load environment --
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[FAIL] GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

print(f"[OK] API key loaded (starts with: {api_key[:10]}...)")

# -- Initialize Gemini client --
from google import genai

client = genai.Client(api_key=api_key)

# -- Test 1: Text Generation with gemini-2.5-flash-lite --
print("\n" + "=" * 60)
print("TEST 1: Text Generation (gemini-2.5-flash-lite)")
print("=" * 60)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="What is Section 302 of the Bharatiya Nyaya Sanhita? Give a brief 2-line answer."
    )
    print(f"[OK] Generation successful!\n")
    print(f"Response:\n{response.text}")
except Exception as e:
    print(f"[FAIL] Generation failed: {e}")

# -- Test 2: Embedding with gemini-embedding-001 --
print("\n" + "=" * 60)
print("TEST 2: Text Embedding (gemini-embedding-001)")
print("=" * 60)

try:
    embedding_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents="Section 302 of the Bharatiya Nyaya Sanhita deals with murder."
    )

    # Extract the embedding vector
    embedding = embedding_response.embeddings[0].values
    print(f"[OK] Embedding successful!")
    print(f"   Dimensions: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"[FAIL] Embedding failed: {e}")

# -- Summary --
print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
print("""
Models configured for this project:
  * Agent LLM:  gemini-2.5-flash-lite  (free tier)
  * Embeddings: gemini-embedding-001   (free tier)

Next step: Run lab/02_data_ingestion.py to load legal datasets.
""")
