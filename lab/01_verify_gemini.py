"""
AI Legal Advisor - Lab Script 01: Verify Gemini API Connection
================================================================
This script validates that:
1. The Gemini API key in .env is working.
2. We can call gemini-3.1-flash-lite for text generation.
3. We can call gemini-embedding-2 for text embeddings.
4. LangGraph + LangChain Google GenAI integration works.

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

# ================================================================
# TEST 1: Text Generation with gemini-3.1-flash-lite (google-genai)
# ================================================================
print("\n" + "=" * 60)
print("TEST 1: Text Generation (gemini-3.1-flash-lite)")
print("=" * 60)

from google import genai

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents="What is Section 302 of the Bharatiya Nyaya Sanhita? Give a brief 2-line answer."
    )
    print(f"[OK] Generation successful!\n")
    print(f"Response:\n{response.text}")
except Exception as e:
    print(f"[FAIL] Generation failed: {e}")

# ================================================================
# TEST 2: Embedding with gemini-embedding-2 (google-genai)
# ================================================================
print("\n" + "=" * 60)
print("TEST 2: Text Embedding (gemini-embedding-2)")
print("=" * 60)

try:
    embedding_response = client.models.embed_content(
        model="gemini-embedding-2",
        contents="Section 302 of the Bharatiya Nyaya Sanhita deals with murder."
    )

    embedding = embedding_response.embeddings[0].values
    print(f"[OK] Embedding successful!")
    print(f"   Dimensions: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"[FAIL] Embedding failed: {e}")

# ================================================================
# TEST 3: LangChain + LangGraph integration with Gemini
# ================================================================
print("\n" + "=" * 60)
print("TEST 3: LangGraph + Gemini Integration")
print("=" * 60)

try:
    os.environ["GOOGLE_API_KEY"] = api_key

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated
    import operator

    # Define a minimal state
    class TestState(TypedDict):
        messages: Annotated[list, operator.add]

    # Initialize Gemini via LangChain
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

    # Define a simple node
    def call_model(state: TestState):
        from langchain_core.messages import HumanMessage
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # Build graph
    workflow = StateGraph(TestState)
    workflow.add_node("agent", call_model)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    app = workflow.compile()

    # Run it
    from langchain_core.messages import HumanMessage
    result = app.invoke({
        "messages": [HumanMessage(content="What is Article 21 of the Indian Constitution? One line only.")]
    })

    final_message = result["messages"][-1]
    print(f"[OK] LangGraph workflow executed successfully!\n")
    print(f"Response:\n{final_message.content}")

except Exception as e:
    print(f"[FAIL] LangGraph test failed: {e}")
    import traceback
    traceback.print_exc()

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
print("""
Models configured for this project:
  * Agent LLM:  gemini-3.1-flash-lite  (free tier)
  * Embeddings: gemini-embedding-2     (free tier, 3072 dims)
  * Framework:  LangGraph + LangChain Google GenAI

Next step: Run lab/02_data_ingestion.py to load legal datasets.
""")
