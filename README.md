# Production Agentic RAG & Routing System

An intelligent, lightweight implementation of **Agentic Retrieval-Augmented Generation (RAG)** built without bulky external agent frameworks. This project demonstrates how reasoning, sub-query decomposition, and tool selection—combined with granular Role-Based Access Control (RBAC)—create an adaptive and secure AI retrieval system.

## 🌟 Overview

Standard RAG pipelines route all requests through static document retrieval paths. **Agentic RAG** introduces agency—giving the system the intelligence to evaluate a query, decompose complex questions into atomic sub-queries, and choose the appropriate domain, tool, or document collection before executing retrieval or generation.

### Key Features
* **Dynamic Query Routing & Sub-Query Decomposition:** Uses an LLM router to break complex user prompts into focused sub-queries and classify requests into OpenAI documentation, 10-K financial filings, or live web search.
* **Vector Search with Qdrant:** Performs semantic search across prebuilt collections (`opnai_data` and `10k_data`) using `nomic-embed-text-v1.5` embeddings.
* **Live Web Retrieval:** Fetches real-time search results via Tavily API for broad or current queries.
* **Role-Based Access Control (RBAC):** Restricts query execution and sub-query routing based on user roles (e.g., `engineer` vs. `finance_analyst`) before vector retrieval or web calls occur.
* **Grounded RAG Generation:** Synthesizes context-aware answers across sub-query results with strict, traceable citation references (`[1]`, `[2]`).

---

## 🏗️ Architecture & Decision Flow

```text
                        ┌─────────────────────┐
                        │     User Query      │
                        └──────────┬──────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Router LLM (GPT)        │
                    │   Sub-Query Decomposition    │
                    │        & route_query()       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      RBAC Permission Check   │
                    │         has_access()         │
                    └──────────────┬───────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │ (Allowed)             │ (Allowed)             │ (Allowed)
           ▼                       ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│    OPENAI_QUERY     │ │ 10K_DOCUMENT_QUERY  │ │   INTERNET_QUERY    │
└──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
           │                       │                       │
           ▼                       ▼                       ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ Nomic Vector Embed  │ │ Nomic Vector Embed  │ │  Tavily Search API  │
└──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
           │                       │                       │
           ▼                       ▼                       │
┌─────────────────────┐ ┌─────────────────────┐           │
│ Qdrant Vector DB    │ │ Qdrant Vector DB    │           │
│ Collection:         │ │ Collection:         │           │
│ "opnai_data"        │ │ "10k_data"          │           │
└──────────┬──────────┘ └──────────┬──────────┘           │
           │                       │                       │
           └───────────┬───────────┘                       │
                       ▼                                   │
           ┌───────────────────────┐                       │
           │    RAG Generator      │                       │
           │  rag_formatted_       │                       │
           │  response()           │                       │
           └───────────┬───────────┘                       │
                       │                                   │
                       └──────────────────┬────────────────┘
                                          ▼
                             ┌────────────────────────┐
                             │     Final Response     │
                             └────────────────────────┘
```
----

## 🔒 Role-Based Access Control (RBAC) Matrix

| Domain / Knowledge Source | Route Label | `engineer` | `finance_analyst` |
| :--- | :--- | :---: | :---: |
| 📘 **OpenAI Documentation** | `OPENAI_QUERY` | ✅ | ✅ |
| 📗 **10-K Filings (Uber/Lyft)** | `10K_DOCUMENT_QUERY` | ❌ | ✅ |
| 🌐 **Live Internet Search** | `INTERNET_QUERY` | ✅ | ❌ |

---

## 🛠️ Prerequisites & Setup

### 1. Dependencies
Install the required packages:

```bash
pip install openai qdrant_client transformers==4.48.0 tavily-python nest_asyncio
```

### Environment Variables
Ensure you have set the following API keys in your environment or Google Colab Secrets:
* `OPENAI_API_KEY`: API key for GPT query decomposition, routing, and RAG generation.
* `TAVILY_API_KEY`: API key for web search querying.

---

## 🚀 Usage

### Basic Unsecured Agentic RAG
Run queries through single-route or multi-query decomposed handlers:

```python
from agentic_rag import agentic_rag, agentic_multi_rag

# Decomposes multi-part financial questions into sub-queries via agentic_multi_rag
agentic_multi_rag("Compare Uber's revenue in 2021 with Lyft's net income")

# Automatically routes broad queries to INTERNET_QUERY via Tavily
agentic_rag("List down new LLMs released in 2025")
```

### Secure Agentic RAG with RBAC
Execute queries with identity enforcement:

```python
from secure_agentic_rag import secure_agentic_rag

# Permitted: bob (finance_analyst) accessing financial data
secure_agentic_rag("bob", "What was Uber's revenue in 2021?")

# Denied: alice (engineer) attempting to access financial data
secure_agentic_rag("alice", "What was Uber's revenue in 2021?")
```

## 📁 Repository Structure

```text
.
├── Agentic_RAG/
│   └── qdrant_data/         # Prebuilt Qdrant vector database files
├── Agentic_Router.ipynb     # Interactive Jupyter/Colab notebook
└── README.md                # Project documentation


