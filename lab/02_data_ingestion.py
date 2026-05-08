"""
AI Legal Advisor - Lab Script 02: Data Ingestion & ChromaDB Setup
==================================================================
This script:
1. Downloads 8 Indian Bare Acts (JSON) from the civictech-India GitHub repo.
2. Parses and normalizes them into a unified schema.
3. Embeds each section using gemini-embedding-2.
4. Stores everything in a local ChromaDB vector database.

Run: .venv\\Scripts\\python lab\\02_data_ingestion.py
"""

import os
import sys
import json
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

# Force unbuffered output on Windows
os.environ["PYTHONUNBUFFERED"] = "1"
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[FAIL] GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

# ================================================================
# STEP 1: Download all available Bare Acts from GitHub
# ================================================================

GITHUB_BASE = "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main"

# Map of filename -> metadata about the Act
ACTS = {
    "ipc.json": {
        "act_name": "Indian Penal Code, 1860",
        "act_short": "IPC",
        "domain": "Criminal",
        "jurisdiction": "Central",
    },
    "crpc.json": {
        "act_name": "Code of Criminal Procedure, 1973",
        "act_short": "CrPC",
        "domain": "Criminal",
        "jurisdiction": "Central",
    },
    "cpc.json": {
        "act_name": "Civil Procedure Code, 1908",
        "act_short": "CPC",
        "domain": "Civil",
        "jurisdiction": "Central",
    },
    "iea.json": {
        "act_name": "Indian Evidence Act, 1872",
        "act_short": "IEA",
        "domain": "Evidence",
        "jurisdiction": "Central",
    },
    "hma.json": {
        "act_name": "Hindu Marriage Act, 1955",
        "act_short": "HMA",
        "domain": "Family",
        "jurisdiction": "Central",
    },
    "ida.json": {
        "act_name": "Indian Divorce Act, 1869",
        "act_short": "IDA",
        "domain": "Family",
        "jurisdiction": "Central",
    },
    "nia.json": {
        "act_name": "Negotiable Instruments Act, 1881",
        "act_short": "NIA",
        "domain": "Commercial",
        "jurisdiction": "Central",
    },
    "MVA.json": {
        "act_name": "Motor Vehicles Act, 1988",
        "act_short": "MVA",
        "domain": "Traffic",
        "jurisdiction": "Central",
    },
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: Downloading Bare Acts from GitHub")
print("=" * 60)

for filename, meta in ACTS.items():
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"  [SKIP] {meta['act_short']} already downloaded.")
        continue

    url = f"{GITHUB_BASE}/{filename}"
    print(f"  [DOWNLOADING] {meta['act_name']}...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"  [OK] Saved to {filepath}")
    except Exception as e:
        print(f"  [FAIL] Could not download {filename}: {e}")

# ================================================================
# STEP 2: Parse and normalize into unified documents
# ================================================================

print("\n" + "=" * 60)
print("STEP 2: Parsing and normalizing all Acts")
print("=" * 60)

all_documents = []

for filename, meta in ACTS.items():
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filename} not found, skipping.")
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            sections = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [FAIL] Could not parse {filename}: {e}")
            continue

    count = 0
    for section in sections:
        # Handle different JSON key formats across datasets
        # Format 1: IPC, CrPC, IEA, NIA (uppercase Section)
        # Format 2: CPC, MVA, IDA (lowercase section/title/description)
        # Format 3: HMA (CSV-like keys)
        sec_num = (
            section.get("Section")
            or section.get("section")
            or section.get("section_number", "Unknown")
        )
        sec_title = (
            section.get("section_title")
            or section.get("title", "")
        )
        sec_desc = (
            section.get("section_desc")
            or section.get("description", "")
        )
        chapter = section.get("chapter", "")
        chapter_title = section.get("chapter_title", "")

        if not sec_desc or not sec_desc.strip():
            continue

        # Build a rich text chunk (Parent-style: includes context)
        text_parts = [meta['act_name']]
        if chapter:
            text_parts.append(f"Chapter {chapter}: {chapter_title}")
        text_parts.append(f"Section {sec_num}: {sec_title}")
        text_parts.append("")
        text_parts.append(sec_desc)
        text = "\n".join(text_parts)

        # Unique ID for ChromaDB
        doc_id = f"{meta['act_short']}_S{sec_num}"

        doc = {
            "id": doc_id,
            "text": text,
            "metadata": {
                "act_name": meta["act_name"],
                "act_short": meta["act_short"],
                "domain": meta["domain"],
                "jurisdiction": meta["jurisdiction"],
                "chapter": str(chapter),
                "chapter_title": str(chapter_title),
                "section": str(sec_num),
                "section_title": str(sec_title),
            },
        }
        all_documents.append(doc)
        count += 1

    print(f"  [OK] {meta['act_short']}: {count} sections parsed.")

print(f"\n  TOTAL: {len(all_documents)} sections across {len(ACTS)} Acts.")

# ================================================================
# STEP 3: Embed and store in ChromaDB
# ================================================================

print("\n" + "=" * 60)
print("STEP 3: Embedding sections and storing in ChromaDB")
print("=" * 60)

from google import genai
import chromadb

client = genai.Client(api_key=api_key)

# Initialize ChromaDB (persistent, stored in data/chromadb/)
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chromadb")
os.makedirs(CHROMA_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Delete old collection and recreate to start clean
try:
    chroma_client.delete_collection("indian_laws")
    print("  [INFO] Deleted old collection to rebuild from scratch.")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="indian_laws",
    metadata={"description": "Indian Bare Acts - Section-level embeddings"}
)

print(f"  [INFO] {len(all_documents)} sections to embed.")

# Batch embed — embed texts individually and collect into batches for ChromaDB
BATCH_SIZE = 20  # Free tier friendly
total_batches = (len(all_documents) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(all_documents))
    batch = all_documents[start:end]

    texts = [d["text"] for d in batch]
    ids = [d["id"] for d in batch]
    metadatas = [d["metadata"] for d in batch]

    try:
        # Embed each text individually to avoid batching issues
        embeddings = []
        for text in texts:
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text,
            )
            embeddings.append(response.embeddings[0].values)

        # Add batch to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"  [OK] Batch {batch_idx + 1}/{total_batches}: embedded {len(batch)} sections. (Total: {end}/{len(all_documents)})")

    except Exception as e:
        print(f"  [FAIL] Batch {batch_idx + 1} failed: {e}")
        print(f"         Waiting 60s before retry (rate limit)...")
        time.sleep(60)
        try:
            embeddings = []
            for text in texts:
                response = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=text,
                )
                embeddings.append(response.embeddings[0].values)

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            print(f"  [OK] Batch {batch_idx + 1}/{total_batches}: retry succeeded.")
        except Exception as e2:
            print(f"  [FAIL] Batch {batch_idx + 1} retry also failed: {e2}")
            print(f"         Skipping this batch. Re-run the script later.")

    # Rate limit: be nice to the free tier
    if batch_idx < total_batches - 1:
        time.sleep(1)

# ================================================================
# STEP 4: Verify the database
# ================================================================

print("\n" + "=" * 60)
print("STEP 4: Verification - Test Query")
print("=" * 60)

# Test: search for "murder"
test_query = "What is the punishment for murder?"

try:
    query_embedding = client.models.embed_content(
        model="gemini-embedding-2",
        contents=test_query,
    )

    results = collection.query(
        query_embeddings=[query_embedding.embeddings[0].values],
        n_results=3,
    )

    print(f"  Query: \"{test_query}\"")
    print(f"  Top 3 results:\n")

    for i, (doc_id, doc_text, metadata) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
    ):
        print(f"  --- Result {i + 1} ---")
        print(f"  ID: {doc_id}")
        print(f"  Act: {metadata['act_name']}")
        print(f"  Section: {metadata['section']} - {metadata['section_title']}")
        print(f"  Text (first 200 chars): {doc_text[:200]}...")
        print()

except Exception as e:
    print(f"  [FAIL] Test query failed: {e}")

# ================================================================
# Summary
# ================================================================

total_in_db = collection.count()
print("=" * 60)
print("DATA INGESTION COMPLETE")
print("=" * 60)
print(f"""
  Database: ChromaDB (persistent at data/chromadb/)
  Collection: indian_laws
  Total sections indexed: {total_in_db}
  Embedding model: gemini-embedding-2 (3072 dims)

  Acts loaded:""")
for meta in ACTS.values():
    print(f"    - {meta['act_name']}")
print(f"""
  Next step: Run lab/03_basic_rag.py to test retrieval + LLM reasoning.
""")
