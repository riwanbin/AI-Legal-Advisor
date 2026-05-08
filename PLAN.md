# AI Legal Advisor — Comprehensive Project Plan

> An AI-powered legal information system for Indian laws that provides structured, citation-backed guidance on real-world legal scenarios.

⚠️ **Disclaimer:** This system provides *legal education and information* only — not formal legal counsel. AI cannot practice law in India under the Advocates Act, 1961.

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Development Philosophy — Lab First](#2-development-philosophy--lab-first)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Agentic Workflow — How the AI Thinks](#4-agentic-workflow--how-the-ai-thinks)
5. [State-of-the-Art RAG Pipeline](#5-state-of-the-art-rag-pipeline)
6. [Multi-Agent Orchestration & Human-in-the-Loop](#6-multi-agent-orchestration--human-in-the-loop)
7. [Dataset Strategy](#7-dataset-strategy)
8. [Data Validation & Maintenance](#8-data-validation--maintenance)
9. [Technology Stack](#9-technology-stack)
10. [Development Phases & Roadmap](#10-development-phases--roadmap)
11. [Example End-to-End Scenario](#11-example-end-to-end-scenario)
12. [Future Scope](#12-future-scope)

---

## 1. Project Vision

A user should be able to describe any real-world legal situation in plain language — a traffic fine, a tenant dispute, a consumer complaint, a workplace issue — and receive:

- **Accurate, grounded advice** citing the exact Acts, Sections, and Clauses that apply.
- **A validity assessment** (e.g., "This fine appears unreasonable because Section X prescribes a maximum of ₹Y").
- **Actionable next steps** (e.g., "You can challenge this via the Consumer Disputes Redressal Forum under the Consumer Protection Act, 2019").
- **A mandatory disclaimer** that this is informational guidance and not a substitute for a qualified advocate.

The system must cover the **full breadth of Indian law** — Criminal, Civil, Constitutional, Corporate, Labour, Consumer, Family, Traffic, Tax, and more — not just one narrow domain.

---

## 2. Development Philosophy — Lab First

We will **NOT** rush into building a web application or API server. The AI reasoning engine is the core product, and it must be battle-tested before any backend wrapping.

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| **Phase 1: Lab** | Workflow, RAG pipeline, agent reasoning | Python scripts, Jupyter notebooks, stable prompts |
| **Phase 2: Backend** | API layer, database persistence | FastAPI server, production vector DB |
| **Phase 3: Frontend** | User-facing interface | Web app (React/Next.js or Streamlit) |

We stay in **Phase 1 (Lab)** until:
- ✅ The AI consistently retrieves the correct legal sections for a query.
- ✅ Zero hallucinations — every claim is grounded in retrieved text.
- ✅ Citations are accurate (correct Act name, Section number, and clause).
- ✅ The clarification loop works (AI asks for missing info instead of guessing).

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
│  "I was fined ₹2000 for parking my bike in Delhi"              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    META-AGENT (LLM)                              │
│  • Analyzes intent                                               │
│  • Identifies legal domain (Traffic, Criminal, Civil, etc.)      │
│  • Decides if it needs more info from user (HITL)                │
│  • Plans which tools to call                                     │
└──────┬───────────┬───────────┬───────────┬──────────────────────┘
       │           │           │           │
       ▼           ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐
│Clarify   │ │Retrieve  │ │Analyze   │ │Cite & Verify     │
│Agent     │ │Agent     │ │Agent     │ │Agent             │
│          │ │          │ │          │ │                  │
│Asks user │ │Hybrid    │ │Applies   │ │Ensures every     │
│for miss- │ │Search +  │ │law to    │ │claim maps to a   │
│ing facts │ │GraphRAG  │ │user's    │ │retrieved source  │
└──────────┘ └──────────┘ │scenario  │ └──────────────────┘
                          └──────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    STRUCTURED RESPONSE                            │
│  • Direct answer with legal assessment                           │
│  • Exact citations (Act, Section, Clause)                        │
│  • Actionable next steps                                         │
│  • Legal disclaimer                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Agentic Workflow — How the AI Thinks

Unlike traditional RAG where retrieval is hardcoded, our system uses an **Agentic** approach: the LLM itself decides what information it needs and which tools to call.

### The Flow:

1. **Intake:** The user describes their scenario in plain language.
2. **Planning:** The LLM Agent analyzes the query and formulates a plan:
   - *"This is a traffic violation. I need the Motor Vehicles Act. But wait — fines vary by state and the user hasn't mentioned their location. I should ask first."*
3. **Clarification (if needed):** The agent pauses and asks the user for missing details.
4. **Tool Calling:** The agent invokes a single, powerful search tool with dynamic filters:
   ```
   search_legal_database(
       query="parking fine two wheeler",
       jurisdiction="Delhi",
       domain="Traffic"
   )
   ```
5. **Multi-hop Reasoning:** If the first retrieval references another section (e.g., "Subject to Section 210..."), the agent automatically calls the tool again to fetch that linked section.
6. **Synthesis:** The agent compiles all retrieved context and formulates the advice.
7. **Citation Verification:** Before responding, a Citation Agent double-checks that every claim maps to a specific retrieved document.
8. **Output:** Structured response with legal assessment, citations, next steps, and disclaimer.

---

## 5. State-of-the-Art RAG Pipeline

Standard "chunk-and-retrieve" RAG is insufficient for law. We implement the latest techniques:

### 5.1 GraphRAG (Knowledge Graphs + Vectors)

**Problem:** Legal documents are deeply interconnected. Section 302 of BNS references the definition of "murder" from Section 101. Pure vector search has no concept of these structural relationships.

**Solution:** Build a Knowledge Graph alongside the vector database.
- **Nodes:** Individual Sections, Acts, Chapters, Schedules.
- **Edges:** "references", "amends", "repeals", "is_exception_to", "is_defined_in".
- **Benefit:** When the agent retrieves Section 302, the graph automatically surfaces all connected sections, giving the LLM the full legal picture.

### 5.2 Contextual Retrieval (Parent-Child Chunking)

**Problem:** Naive fixed-size chunking (e.g., 500 tokens) often splits a legal section in half, destroying its meaning.

**Solution:** Two-tier chunking strategy:
- **Child Chunks (for search):** Small, precise chunks — ideally one sub-section or clause. These are embedded and indexed for maximum search accuracy.
- **Parent Chunks (for context):** When a child chunk is matched, we retrieve the entire parent (the full Section or even the full Chapter) and pass it to the LLM so it reads the law in its complete context.

### 5.3 Hybrid Search with Neural Re-ranking

**Problem:** Vector search is great for semantic similarity but can miss exact legal terms like "Section 498A" or "Habeas Corpus". Keyword search finds these but lacks semantic understanding.

**Solution:** Combine both and re-rank:
1. **BM25 Keyword Search** — catches exact terms, section numbers, legal jargon.
2. **Dense Vector Search** — catches semantic meaning ("Can my landlord evict me?" → matches tenancy laws).
3. **Neural Re-ranker** (e.g., Cohere Rerank, ColBERT, or a cross-encoder) — takes the merged results and re-orders them by true relevance to the query.

### 5.4 Query Transformation (HyDE)

**Problem:** A user's casual query ("Can my boss fire me for no reason?") is very different in language from the actual legal text ("Termination of employment without cause under the Industrial Disputes Act...").

**Solution:** Use **Hypothetical Document Embeddings (HyDE)**:
1. The LLM first generates a "hypothetical ideal legal answer" to the user's query.
2. We embed *that* hypothetical answer and search the vector database with it.
3. This dramatically improves retrieval accuracy because the search query now "speaks the same language" as the legal documents.

---

## 6. Multi-Agent Orchestration & Human-in-the-Loop

### Sub-Agents

| Agent | Role | When It Runs |
|-------|------|--------------|
| **Clarification Agent** | Detects missing critical facts (state, date, parties involved) and asks the user | Before retrieval, if the query is ambiguous |
| **Retrieval Agent** | Formulates optimized search queries, applies filters, performs multi-hop retrieval | After the query is understood |
| **Analysis Agent** | Reads retrieved law and applies it to the user's specific facts | After documents are retrieved |
| **Citation Agent** | Validates that every factual claim in the response has a grounded source | Before the final response is sent |

### Human-in-the-Loop (HITL) — Clarification Flow

Many legal questions cannot be answered without context. Instead of guessing or hallucinating, the agent **pauses and asks**:

```
User: "I got a challan, is the amount correct?"

Agent (thinking): "I don't know:
  - What type of violation?
  - Which state/city?
  - What amount was charged?
  I must ask the user before proceeding."

Agent → User: "I'd like to help you verify your challan. Could you please share:
  1. What was the violation mentioned on the challan?
  2. Which city/state did this happen in?
  3. What is the fine amount charged?"

User: "It was for not wearing a helmet in Bangalore, ₹1000."

Agent (resumes): Now searches Karnataka Traffic Police penalty chart
  + Motor Vehicles Act Section 129...
```

This is implemented using **LangGraph's interrupt/resume** mechanism or a similar stateful workflow engine.

---

## 7. Dataset Strategy

### 7.1 Scope — All Major Domains of Indian Law

| Domain | Key Acts |
|--------|----------|
| **Criminal** | Bharatiya Nyaya Sanhita (BNS), BNSS, BSA, NDPS Act, POCSO Act |
| **Civil** | Civil Procedure Code (CPC), Limitation Act, Specific Relief Act |
| **Constitutional** | Constitution of India (All Articles + Amendments) |
| **Family** | Hindu Marriage Act, Special Marriage Act, Hindu Succession Act, Muslim Personal Law |
| **Consumer** | Consumer Protection Act, 2019 |
| **Corporate** | Companies Act 2013, LLP Act, SEBI Act, Insolvency & Bankruptcy Code |
| **Labour** | Industrial Disputes Act, Factories Act, new Labour Codes (2020) |
| **Tax** | Income Tax Act, GST Acts (CGST, SGST, IGST) |
| **Property** | Transfer of Property Act, Registration Act, RERA |
| **Traffic** | Motor Vehicles Act 1988 (with 2019 Amendment) |
| **Cyber & IT** | Information Technology Act, 2000 (with 2008 Amendment) |
| **State-Specific** | State Traffic Penalties, Rent Control Acts, Municipal By-Laws |

### 7.2 Data Sources

| Source | Type | What It Provides |
|--------|------|-----------------|
| **Hugging Face — OpenNYAI** | Pre-cleaned dataset | Indian statutes, case law, structured for AI |
| **Kaggle** | CSV/JSON datasets | IPC sections, BNS, Supreme Court judgments |
| **IndiaCode (`indiacode.nic.in`)** | Official government portal | All Central and State Acts (PDF/XML) — latest amendments |
| **Indian Kanoon** | Case law repository | Court judgments and precedents |

### 7.3 Data Processing Pipeline

```
Raw Data (PDF/CSV/XML)
    │
    ▼
┌───────────────────────┐
│ 1. Parse & Clean      │  Extract structured text from PDFs/XMLs
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 2. Section-Level      │  Split text into individual Sections/Articles
│    Chunking           │  (NOT arbitrary token windows)
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 3. Metadata Tagging   │  Add: Act, Chapter, Section No., Domain,
│                       │  Jurisdiction (Central/State), Amendment Date
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 4. Relationship       │  Extract cross-references ("Subject to Sec 10")
│    Extraction         │  to build Knowledge Graph edges
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 5. Embed & Store      │  • Child chunks → Vector DB (ChromaDB/Qdrant)
│                       │  • Parent chunks → Document Store
│                       │  • Relationships → Knowledge Graph (Neo4j)
└───────────────────────┘
```

---

## 8. Data Validation & Maintenance

### How do we know what's missing or outdated?

| Mechanism | How It Works |
|-----------|-------------|
| **Master Index Cross-Reference** | IndiaCode publishes an "Alphabetical List of Central Acts." A periodic script downloads this list, compares it against our database, and flags any Acts we don't have. |
| **Amendment Date Tracking** | Every section in our DB is tagged with `last_amended_date`. A scheduled job checks IndiaCode for newer amendment dates and triggers a re-ingestion if found. |
| **Web-Search Fallback (Live)** | The Agent has a `search_web()` tool. Before finalizing critical advice, it can do a quick web search to check for very recent ordinances or Supreme Court stays that haven't reached our database yet. |
| **Coverage Dashboard** | A simple script that reports: total Acts ingested, total Sections indexed, last sync date, and any known gaps. |

---

## 9. Technology Stack

### Phase 1 — Lab (Current Focus)

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Environment** | Python 3.11+ / Jupyter Notebooks | Rapid iteration, easy debugging |
| **Agent Framework** | LangGraph | Best-in-class for stateful multi-agent workflows with HITL support |
| **RAG Framework** | LlamaIndex | Industry leader for advanced retrieval (GraphRAG, Parent-Child, Hybrid Search) |
| **Vector Database** | ChromaDB (local) | Zero-config, perfect for lab experimentation |
| **Knowledge Graph** | Neo4j (local via Docker) | For GraphRAG — mapping legal cross-references |
| **LLM (Agent)** | `gemini-2.5-flash-lite` (Gemini free tier) | Fast, cost-free, good reasoning for lab iteration |
| **Embedding Model** | `gemini-embedding-001` (Gemini free tier, 3072 dims) | High-quality text embeddings for legal retrieval |
| **Re-ranker** | Cohere Rerank / Cross-Encoder | For Hybrid Search result re-ordering |

### Phase 2 — Backend (After Lab Stabilization)

| Component | Technology |
|-----------|------------|
| **API Framework** | FastAPI |
| **Vector DB (Production)** | Qdrant or Pinecone |
| **Task Queue** | Celery + Redis (for async processing) |
| **Containerization** | Docker + Docker Compose |

### Phase 3 — Frontend (After Backend)

| Component | Technology |
|-----------|------------|
| **Web Framework** | React / Next.js or Streamlit |
| **Chat Interface** | Custom chat UI with citation rendering |

---

## 10. Development Phases & Roadmap

### Phase 1: Lab — Core AI Engine (Current)

- [ ] **1.1** Initialize project structure and virtual environment.
- [ ] **1.2** Download a starter dataset (IPC/BNS from Kaggle or Hugging Face).
- [ ] **1.3** Build the data processing pipeline (parse → chunk by section → tag metadata).
- [ ] **1.4** Set up ChromaDB and test basic vector search.
- [ ] **1.5** Implement Parent-Child chunking and test contextual retrieval.
- [ ] **1.6** Add BM25 keyword search alongside vector search (Hybrid Search).
- [ ] **1.7** Integrate a Neural Re-ranker.
- [ ] **1.8** Build the Knowledge Graph in Neo4j (cross-reference extraction).
- [ ] **1.9** Build the LangGraph multi-agent workflow (Meta → Clarification → Retrieval → Analysis → Citation).
- [ ] **1.10** Implement Human-in-the-Loop clarification flow.
- [ ] **1.11** Test with 20+ diverse legal scenarios across different domains.
- [ ] **1.12** Evaluate: retrieval accuracy, hallucination rate, citation correctness.

### Phase 2: Backend — API Layer

- [ ] **2.1** Wrap the stable lab engine in a FastAPI application.
- [ ] **2.2** Migrate to production vector database (Qdrant).
- [ ] **2.3** Add session management for multi-turn conversations.
- [ ] **2.4** Build the data maintenance CRON jobs (amendment checking, coverage reports).
- [ ] **2.5** Containerize with Docker.

### Phase 3: Frontend — User Interface

- [ ] **3.1** Build a chat-style web interface.
- [ ] **3.2** Render citations as clickable links to the original law text.
- [ ] **3.3** Add a "Verify this advice" button for user feedback loops.

---

## 11. Example End-to-End Scenario

**User Query:** *"My landlord is refusing to return my security deposit of ₹50,000 after I vacated the flat in Mumbai. What can I do?"*

**Agent Workflow:**

1. **Meta-Agent** identifies: Domain = Tenancy/Property, Location = Mumbai (Maharashtra).
2. **Clarification Agent** asks: *"How long ago did you vacate? Was there a written rental agreement?"*
3. **User responds:** *"3 months ago. Yes, there was a registered agreement."*
4. **Retrieval Agent** searches:
   - `search_legal_database(query="security deposit refusal", jurisdiction="Maharashtra", domain="Tenancy")`
   - Retrieves: Maharashtra Rent Control Act, 1999 + Transfer of Property Act provisions.
5. **Analysis Agent** applies the law:
   - Under the agreement terms + Maharashtra Rent Control Act, the landlord must return the deposit within a reasonable period after deducting legitimate damages.
   - If no damages, full refund is owed.
6. **Citation Agent** verifies all claims are grounded.
7. **Output:**
   > **Legal Assessment:** Under the Maharashtra Rent Control Act, 1999 (Section X), and your registered rental agreement, your landlord is obligated to return the security deposit minus any legitimate deductions for damages. A 3-month delay without communication may constitute unreasonable withholding.
   >
   > **Applicable Laws:** Maharashtra Rent Control Act, 1999, Section X; Transfer of Property Act, 1882, Section 108(m).
   >
   > **Recommended Next Steps:**
   > 1. Send a formal legal notice via registered post demanding the return of ₹50,000 within 15 days.
   > 2. If unresolved, file a complaint with the Rent Court or the Consumer Disputes Redressal Forum under the Consumer Protection Act, 2019.
   >
   > ⚠️ *This is AI-generated informational guidance. Please consult a qualified advocate for formal legal advice.*

---

## 12. Future Scope

- **Multilingual Support:** Hindi, Marathi, Tamil, Bengali — most citizens don't speak English.
- **Voice Input:** Allow users to describe their legal situation verbally (Speech-to-Text).
- **Document Analysis:** Upload a legal notice or FIR and get an AI-powered summary and analysis.
- **Lawyer Referral Integration:** If the situation is complex, connect the user with a verified advocate in their city.
- **Case Law Prediction:** Based on similar past cases and judgments, estimate the likely outcome.

---

*Last updated: May 2026*
