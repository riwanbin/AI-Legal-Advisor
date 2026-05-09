"""
AI Legal Advisor - Lab Script 03: Basic RAG & Agent Workflow
================================================================
This script implements a stateful, multi-agent RAG workflow using LangGraph.
It connects the ChromaDB vector database (populated in lab 02) with 
the gemini-3.1-flash-lite LLM.

Agents:
1. Clarification Node: Detects if the query lacks necessary facts and asks the user.
2. Retrieval Node: Queries ChromaDB for relevant legal sections.
3. Analysis Node: Generates a final answer based on retrieved documents.
"""

import os
import sys
import operator
from typing import TypedDict, Annotated, List

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
import chromadb
from google import genai

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# ================================================================
# SETUP
# ================================================================

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[FAIL] GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = api_key

# 1. Initialize Gemini Client for Embeddings
client = genai.Client(api_key=api_key)

# 2. Initialize ChromaDB
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chromadb")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    collection = chroma_client.get_collection(name="indian_laws")
    print(f"[OK] Connected to ChromaDB. Indexed sections: {collection.count()}")
except Exception as e:
    print(f"[FAIL] Could not load ChromaDB collection 'indian_laws'. Did you run lab 02?\nError: {e}")
    sys.exit(1)

# 3. Initialize LangChain LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)

# ================================================================
# LANGGRAPH DEFINITIONS
# ================================================================

# Define the State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    context: str
    clarification_needed: bool

# ----------------------------------------------------------------
# Node 1: Clarification Agent
# ----------------------------------------------------------------
def clarification_node(state: AgentState):
    """
    Checks if the user's latest message has enough context to answer legally.
    If it's too vague, asks for more details.
    """
    print("\n--- [Clarification Agent Thinking] ---")
    
    system_prompt = """You are an expert Indian Legal Clarification Agent.
Your job is to read the user's legal query and decide if you have enough facts to perform a targeted legal search.
If the query is extremely vague, hypothetical without context, or missing critical details (like the state/city, specific amounts, or the exact nature of the violation), you MUST ask a clarifying question.
If the query is a straightforward legal question (e.g., "What is the punishment for murder?" or "Define dowry"), DO NOT ask for clarification.
If the user provides the missing details after you asked, DO NOT ask again.

Respond strictly in the following format:
If you need clarification, start your response with "CLARIFICATION_NEEDED: " followed by your question to the user.
If the query is clear enough, output exactly "PROCEED".
"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    
    content = response.content
    if isinstance(content, list):
        # Sometime LangChain returns a list of blocks for multimodal models
        content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        
    content = content.strip()
    
    if content.startswith("CLARIFICATION_NEEDED:"):
        question = content.replace("CLARIFICATION_NEEDED:", "").strip()
        print(f"Decision: Needs more info. Asking: {question}")
        return {
            "messages": [AIMessage(content=question)],
            "clarification_needed": True
        }
    else:
        print("Decision: Query is clear enough. Proceeding to retrieval.")
        return {
            "clarification_needed": False
        }

# ----------------------------------------------------------------
# Node 2: Retrieval Agent
# ----------------------------------------------------------------
def retrieval_node(state: AgentState):
    """
    Formulates a search query, embeds it, and fetches relevant documents.
    """
    print("\n--- [Retrieval Agent Searching] ---")
    
    # Simple approach: just use the latest user message as the search query.
    # A more advanced approach would use the LLM to generate a search query from the chat history.
    latest_message = state["messages"][-1].content
    
    print(f"Embedding query: '{latest_message}'")
    
    # 1. Embed the query
    query_embedding = client.models.embed_content(
        model="gemini-embedding-2",
        contents=latest_message,
    )
    
    # 2. Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.embeddings[0].values],
        n_results=4, # Fetch top 4 most relevant sections
    )
    
    # 3. Format the context
    context_parts = []
    print("\nFound relevant sections:")
    for doc_id, doc_text, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
        act_name = metadata.get("act_name", "Unknown Act")
        section = metadata.get("section", "Unknown Section")
        print(f" - {act_name}, Section {section} (ID: {doc_id})")
        
        context_parts.append(f"--- Document ID: {doc_id} ---\n{doc_text}\n")
        
    context_str = "\n".join(context_parts)
    
    return {
        "context": context_str
    }

# ----------------------------------------------------------------
# Node 3: Analysis Agent
# ----------------------------------------------------------------
def analysis_node(state: AgentState):
    """
    Reads the retrieved context and answers the user's question.
    """
    print("\n--- [Analysis Agent Generating Response] ---")
    
    system_prompt = f"""You are an expert Indian Legal AI Advisor.
Your task is to answer the user's question based strictly on the provided legal documents below.
Do not hallucinate legal facts. If the documents do not contain the answer, say that you don't have enough information in your current database.
Always cite the Act name and Section number when providing an answer.

<documents>
{state["context"]}
</documents>
"""
    
    # Replace the actual system message if we wanted, but here we just append our context-aware instruction
    # to the current conversation to get the final answer.
    # We create a temporary message list just for this invocation so we don't pollute the actual state history
    # with the giant context block permanently.
    
    invoke_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(invoke_messages)
    
    content = response.content
    if isinstance(content, list):
        content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        
    return {
        "messages": [AIMessage(content=content)]
    }

# ================================================================
# BUILD THE GRAPH
# ================================================================

def should_clarify(state: AgentState):
    if state.get("clarification_needed"):
        return "end" # Suspend execution and wait for user input
    return "retrieve"

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("clarify", clarification_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("analyze", analysis_node)

# Add edges
workflow.set_entry_point("clarify")
workflow.add_conditional_edges("clarify", should_clarify, {
    "end": END,
    "retrieve": "retrieve"
})
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", END)

# Compile the graph
app = workflow.compile()

# ================================================================
# INTERACTIVE REPL
# ================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Welcome to the AI Legal Advisor (Lab 03 - Basic RAG)")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    # We maintain the conversation state manually in the REPL
    chat_state = {"messages": [], "context": "", "clarification_needed": False}
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
            
        if not user_input:
            continue
            
        # Add user message to state
        chat_state["messages"].append(HumanMessage(content=user_input))
        
        try:
            # Run the graph
            result_state = app.invoke(chat_state)
            
            # Update our local tracking state
            chat_state = result_state
            
            # The last message is the agent's response
            final_response = chat_state["messages"][-1].content
            
            print(f"\nAgent: {final_response}")
            
        except Exception as e:
            print(f"\n[ERROR] Graph execution failed: {e}")
            import traceback
            traceback.print_exc()
