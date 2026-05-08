# AI Legal Advisor (Indian Laws)

An AI-powered legal information system that provides structured, citation-backed guidance for real-world legal scenarios under Indian law.

> ⚠️ **Disclaimer:** This system provides legal *education and information* only — not formal legal advice. AI cannot practice law in India under the Advocates Act, 1961.

## What This Project Does

A user describes any legal situation in plain language — a traffic fine, a tenant dispute, a consumer complaint — and the AI:

- Identifies the applicable Indian laws (Acts, Sections, Clauses)
- Assesses the situation (e.g., "Your fine appears unreasonable")
- Provides actionable next steps with proper legal citations
- Asks for clarification when critical details are missing

## Architecture

Built on a **State-of-the-Art RAG pipeline** with:

- **GraphRAG** — Knowledge Graphs mapping relationships between Acts and Sections
- **Parent-Child Chunking** — Precise retrieval with full contextual reading
- **Hybrid Search + Neural Re-ranking** — BM25 + Vector Search + Cohere Rerank
- **Multi-Agent Orchestration** — Specialized agents for retrieval, analysis, citation verification
- **Human-in-the-Loop** — AI asks the user for missing context instead of guessing

## Development Approach

**Lab First** — We stabilize the AI reasoning engine in Python notebooks before building any backend or frontend.

See [PLAN.md](./PLAN.md) for the full comprehensive project plan.

## Current Status

🔬 **Phase 1: Lab** — Building and testing the core AI engine.

## Tech Stack (Lab Phase)

| Component | Technology |
|-----------|------------|
| Agent Framework | LangGraph |
| RAG Framework | LlamaIndex |
| Vector DB | ChromaDB |
| Knowledge Graph | Neo4j |
| LLM | Gemini 1.5 Pro / GPT-4o |

## License

TBD
