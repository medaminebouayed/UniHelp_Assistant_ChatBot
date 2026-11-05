# UniHelp — Intelligent Hybrid Graph RAG Academic Platform (IIT & ISB)

UniHelp is a full-stack, enterprise-grade academic assistant and administration platform designed for Tunisian higher education institutions: **IIT** (*Institut International de Technologie / Institut des Ingénieurs de Tunis*) and **ISB** (*International School of Business*).

The platform leverages an advanced **Hybrid Graph Retrieval-Augmented Generation (Graph RAG)** architecture combining dense semantic vectors, sparse lexical BM25 matching, knowledge graph traversal (Neo4j), Reciprocal Rank Fusion (RRF), and cross-encoder neural reranking to accurately answer complex student queries, automate administrative email generation, and streamline university administrative workflows.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [RAG Core & Retrieval Pipeline](#rag-core--retrieval-pipeline)
  - [1. Data Processing \& Cleaning](#1-data-processing--cleaning)
  - [2. Semantic Chunking \& Augmentation](#2-semantic-chunking--augmentation)
  - [3. Knowledge Graph Engine (Neo4j)](#3-knowledge-graph-engine-neo4j)
  - [4. Vector \& Lexical Storage (pgvector + BM25)](#4-vector--lexical-storage-pgvector--bm25)
  - [5. Multi-Stage Hybrid Retrieval & Fusion](#5-multi-stage-hybrid-retrieval--fusion)
  - [6. Cross-Encoder Neural Reranking](#6-cross-encoder-neural-reranking)
  - [7. Answer Generation & Citations](#7-answer-generation--citations)
- [Backend Services (FastAPI)](#backend-services-fastapi)
- [Frontend Application (Angular 17+)](#frontend-application-angular-17)
- [Database Models & Schemas](#database-models--schemas)
- [Prerequisites & System Requirements](#prerequisites--system-requirements)
- [Installation & Quick Start](#installation--quick-start)
  - [Option A: Docker Compose (Recommended)](#option-a-docker-compose-recommended)
  - [Option B: Local Development Scripts (Windows)](#option-b-local-development-scripts-windows)
- [Environment Configuration](#environment-configuration)
- [API Endpoints Reference](#api-endpoints-reference)
- [Testing & Evaluation](#testing--evaluation)

---

## Key Features

### For Students:
- **Intelligent RAG Assistant**: Ask questions in French or English regarding curricula, tuition fees, dual degrees (e.g., 3iL, ISCOM, Audencia), accreditations (EUR-ACE), admission procedures, mobility, and syllabi.
- **Source Verification & Citations**: Direct reference to official institution documents, scraped web portals, and catalogs.
- **Administrative Email Assistant**: Automated generator producing professional administrative requests (transcripts, internship agreements, attestations, enrollment inquiries) matching official institutional standards.
- **Conversational Memory**: Multi-session persistent chat histories.
- **Interactive FAQ**: Dynamic frequently asked questions categorized by department and site.

### For Administrators:
- **Comprehensive Analytics Dashboard**: Real-time stats on conversations, active students, most frequently asked topics, and document coverage.
- **Document Ingestion & Live Re-indexing**: Upload PDFs, DOCX, or text files directly via the dashboard; documents are automatically cleaned, chunked, embedded into PostgreSQL pgvector, synchronized into Neo4j graph nodes, and incorporated into BM25.
- **Student & Staff Management**: View student profiles, registration statuses, and manage access rights.
- **Dual Site Segregation**: Strict domain-specific separation and filtering between **IIT** (Engineering) and **ISB** (Business).

---

## System Architecture

```
                                  ┌────────────────────────┐
                                  │      Web Browser       │
                                  │ (Angular 17+ / Tailwind)│
                                  └───────────┬────────────┘
                                              │ HTTP / REST
                                              ▼
                                  ┌────────────────────────┐
                                  │   FastAPI Web API      │
                                  │      (Port 8000)       │
                                  │ • Auth & JWT           │
                                  │ • RBAC (Student/Admin) │
                                  │ • History & Management │
                                  └───────────┬────────────┘
                                              │ HTTP Internal Proxy
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               RAG Microservice (Port 8002)                             │
│                                                                                        │
│   ┌───────────────────┐    ┌────────────────────┐    ┌─────────────────────────────┐   │
│   │ Dense Search      │    │ Sparse BM25 Search │    │ Graph Entity Search (Neo4j) │   │
│   │ (pgvector cosine) │    │ (French Stemming)  │    │ (Cypher 1-2 Hop Traversal)  │   │
│   └─────────┬─────────┘    └─────────┬──────────┘    └──────────────┬──────────────┘   │
│             │                        │                              │                  │
│             └─────────────────┐      │      ┌───────────────────────┘                  │
│                               ▼      ▼      ▼                                          │
│                       ┌──────────────────────────────┐                                 │
│                       │ Weighted RRF (Rank Fusion)   │                                 │
│                       └──────────────┬───────────────┘                                 │
│                                      ▼                                                 │
│                       ┌──────────────────────────────┐                                 │
│                       │ Cross-Encoder Neural Rerank  │                                 │
│                       │ (BAAI/bge-reranker-base)     │                                 │
│                       └──────────────┬───────────────┘                                 │
│                                      ▼                                                 │
│                       ┌──────────────────────────────┐                                 │
│                       │ Prompt Engineering + Context │                                 │
│                       └──────────────┬───────────────┘                                 │
│                                      │                                                 │
│                     ┌────────────────┴────────────────┐                                │
│                     ▼                                 ▼                                │
│         ┌───────────────────────┐         ┌───────────────────────────┐                │
│         │ Local Ollama Service  │         │ External Groq Cloud API   │                │
│         │ (Qwen2.5:7b / 32b)    │   OR    │ (Qwen / Llama-3.1-8b)     │                │
│         └───────────────────────┘         └───────────────────────────┘                │
└────────────────────────────────────────────────────────────────────────────────────────┘
            │                                                      │
            ▼                                                      ▼
 ┌──────────────────────┐                               ┌──────────────────────┐
 │ PostgreSQL + pgvector│                               │     Neo4j DBMS       │
 │   - Metadata & Users │                               │   - Programs & Etab  │
 │   - Dense Embeddings │                               │   - Diplomas & Units │
 │   - 384-dim Vectors  │                               │   - Graph Relations  │
 └──────────────────────┘                               └──────────────────────┘
```

---

## Repository Structure

```
.
├── backend/                  # FastAPI Core Application Service (Port 8000)
│   ├── app/
│   │   ├── models/           # SQLAlchemy DB models (User, Student, Admin, Conversation, Message, Doc)
│   │   ├── routers/          # API Route handlers (auth, chat, student, admin, dashboard, etc.)
│   │   ├── schemas/          # Pydantic v2 validation schemas
│   │   ├── utils/            # JWT security, password hashing, mailer, RAG proxy helpers
│   │   └── database.py       # SQLAlchemy engine & session manager
│   ├── uploads/              # Storage directory for uploaded documents
│   ├── Dockerfile            # Container configuration for backend API
│   ├── main.py               # Application entrypoint & lifespan migrations
│   └── requirements.txt      # Python backend dependencies
│
├── frontend/                 # Modern Angular 17+ Client (Port 4200 / 80 in Docker)
│   ├── src/
│   │   ├── app/
│   │   │   ├── components/   # Chatbot floating FAB, branding logos, reusable widgets
│   │   │   ├── guards/       # Role-based Angular route guards
│   │   │   ├── interceptors/ # Bearer JWT token injection interceptor
│   │   │   ├── layouts/      # StudentLayout, AdminLayout, AuthLayout
│   │   │   ├── pages/        # Dashboard, Chatbot, Documents, Profile, Settings, FAQ, Email Gen
│   │   │   └── services/     # Angular HTTP services interfacing with Backend
│   │   ├── environments/     # API URL environment profiles
│   │   └── styles.css        # Tailwind CSS and global style definitions
│   ├── nginx.conf            # Production Nginx reverse proxy configuration
│   └── package.json          # Node.js dependencies
│
├── RAG Core/                 # Dedicated RAG Architecture & Microservice (Port 8002)
│   ├── rag/                  # Core RAG engine modules
│   │   ├── config.py         # Hyperparameters (weights, top-k, thresholds, connection URIs)
│   │   ├── dense_index.py    # PostgreSQL pgvector similarity queries
│   │   ├── graph_index.py    # Neo4j Cypher extraction and path traversals
│   │   ├── sparse_index.py   # BM25Okapi inverted index search with French tokenization
│   │   ├── models.py         # Embedding (MiniLM) and Cross-Encoder model caching
│   │   ├── retriever.py      # Multi-index retrieval with Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py       # Neural cross-encoder reranking engine
│   │   ├── ingest.py         # Document parsing, normalization, and insertion
│   │   └── synthetic.py      # LLM-based synthetic Q&A generation for chunks
│   ├── rag_api/              # Standalone FastAPI microservice for RAG operations
│   │   └── main.py           # Endpoints: /process, /index-unindexed, /query, /generate-email
│   ├── utils/                # Offline processing, evaluation, and data pipeline scripts
│   │   ├── augmentation/     # Metadata enrichment, Q&A synthesis
│   │   ├── evaluation/       # RRF grid search, HyDE, and retrieval benchmarking
│   │   ├── graph/            # Knowledge graph population (kge_engine.py)
│   │   └── preprocessing/    # Cleaning, deduplication, and semantic chunking
│   ├── data/                 # Serialized BM25 indexes, checkpoints, evaluation datasets
│   └── Dockerfile            # Container configuration for RAG engine (NVIDIA GPU ready)
│
├── tests/                    # End-to-end and unit test suite
│   ├── test_rag_performance.py
│   ├── test_api_full.py
│   └── debug_rerank.py
│
├── docker-compose.yml        # Orchestration for PostgreSQL (pgvector), Neo4j, Ollama, Backend, RAG API, Frontend
├── start_services.bat        # Windows one-click local development launcher
├── stop_services.bat         # Windows service termination script
└── db_migrate.py             # Schema migration utility
```

---

## RAG Core & Retrieval Pipeline

The heart of UniHelp is an advanced hybrid information retrieval system designed to avoid hallucinations on specialized educational curricula:

### 1. Data Processing & Cleaning
- **Raw Sources**: Over 2,100 curated documents and chunks scraped from official IIT and ISB web portals, internal student guides, program brochures, and syllabi.
- **Normalization**: French diacritic sanitization, header/footer stripping, boilerplate removal, and metadata tagging (institution, category, module code).

### 2. Semantic Chunking & Augmentation
- **Chunking Strategy**: Adaptive sliding-window semantic chunking (30–200 tokens, average 162 tokens) preventing fragmented table data or disrupted curriculum requirements.
- **Augmentation**: Every chunk is enhanced with synthetic query-answer pairs and metadata keywords to increase lexical and semantic recall during retrieval.

### 3. Knowledge Graph Engine (Neo4j)
- Extracts named entities and formal relationships:
  - `(:Institution {name: "IIT"})`
  - `(:Program {name: "Génie Informatique"})-[:OFFERED_BY]->(:Institution)`
  - `(:Program)-[:HAS_DOUBLE_DEGREE]->(:PartnerUniversity {country: "France"})`
  - `(:Module)-[:TAUGHT_IN]->(:Semester)`
  - `(:Chunk)-[:MENTIONS]->(:Entity)`
- Enables 1-hop and 2-hop Cypher queries to answer relational multi-hop questions that vector similarity alone cannot resolve.

### 4. Vector & Lexical Storage (pgvector + BM25)
- **Dense Vectors**: Chunks embedded using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and stored inside PostgreSQL with the `pgvector` extension using HNSW/cosine indexing.
- **Sparse Index**: Lexical index powered by `rank-bm25` (BM25Okapi) with customized French stopword filtering, accent normalization, and morphological query expansion.

### 5. Multi-Stage Hybrid Retrieval & Fusion
When a user asks a question:
1. **Parallel Fetch**: 
   - Top-$K$ dense vector matches from PostgreSQL pgvector.
   - Top-$K$ lexical matches from BM25 index.
   - Connected entities and linked chunks from Neo4j Graph.
2. **Weighted Reciprocal Rank Fusion (RRF)**:
   $$\text{RRF\_Score}(d) = w_{\text{dense}} \cdot \frac{1}{k + r_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{k + r_{\text{sparse}}(d)} + w_{\text{graph}} \cdot \frac{1}{k + r_{\text{graph}}(d)}$$
   *(Default configuration: $k=60$, $w_{\text{dense}}=0.6$, $w_{\text{sparse}}=0.4$)*.

### 6. Cross-Encoder Neural Reranking
- The top-25 merged candidates from RRF pass through a cross-encoder model: **`BAAI/bge-reranker-base`**.
- It evaluates the full joint token interaction between `(query, document_chunk)` and re-scores candidates with high precision.
- Keeps only the top-5 to top-7 highest scoring chunks.

### 7. Answer Generation & Citations
- **Inference Engine**: Dispatched to **Ollama** (`qwen2.5:7b` / `qwen2.5:32b`) locally or accelerated via **Groq Cloud** API (`llama-3.1-8b-instant`).
- **Prompt Guardrails**: Strictly grounded context-only answering with inline bracket citations `[Source X]` and structured markdown tables for module listings, tuition fees, and admission criteria.

---

## Backend Services (FastAPI)

Built with **FastAPI** adhering to clean layered architecture:
- **`app/models`**: Relational ORM models utilizing SQLAlchemy (PostgreSQL).
- **`app/schemas`**: Strict request/response validation with Pydantic v2.
- **`app/routers`**:
  - `auth`: JWT generation, bearer authentication, password hashing with bcrypt.
  - `chat`: Conversational proxy between students and the RAG engine.
  - `conversation`: Session history, renaming, deletion, message persistence.
  - `document`: File uploads, byte storage, indexing triggers, chunk management.
  - `email_genere`: Generation and history tracking of formal administrative emails.
  - `dashboard`: Statistical aggregation for administrative visibility.
  - `etudiant` & `administrateur`: Profile and user administration.

---

## Frontend Application (Angular 17+)

The UI is a standalone Single Page Application (SPA) built with **Angular 17+** and **Tailwind CSS**:
- **Design System**: Responsive layout supporting dark and light themes via Tailwind.
- **Student Portal**:
  - Interactive chat interface with real-time markdown formatting, source pills, and copy utilities.
  - Floating action button (FAB) for quick access across pages.
  - Dedicated administrative email drafting wizard with instant clipboard export.
  - Searchable institution FAQ.
- **Admin Portal**:
  - Real-time KPI cards (active users, total queries, documents processed).
  - Document management data table with direct upload, delete, and manual re-index triggers.
  - Student registry and audit tables.
- **Interceptors & Guards**: Automatic HTTP `Authorization: Bearer <token>` injection and route protection by user role.

---

## Database Models & Schemas

### Relational Schema (PostgreSQL)

```mermaid
erDiagram
    UTILISATEUR ||--o| ETUDIANT : "is a"
    UTILISATEUR ||--o| ADMINISTRATEUR : "is a"
    ETUDIANT ||--o{ CONVERSATION : "owns"
    CONVERSATION ||--o{ MESSAGE : "contains"
    ETUDIANT ||--o{ EMAIL_GENERE : "creates"
    ADMINISTRATEUR ||--o{ DOCUMENT : "manages"

    UTILISATEUR {
        int id PK
        string nom
        string prenom
        string email UK
        string mot_de_passe
        string role
        datetime date_creation
    }

    ETUDIANT {
        int id PK, FK
        string filiere
        string niveau
        string universite
        string telephone
    }

    ADMINISTRATEUR {
        int id PK, FK
        string departement
        string telephone
    }

    CONVERSATION {
        int id PK
        int etudiant_id FK
        string titre
        datetime date_creation
        datetime date_modification
    }

    MESSAGE {
        int id PK
        int conversation_id FK
        string expediteur
        string contenu
        jsonb sources
        datetime date_envoi
    }

    DOCUMENT {
        int id PK
        int admin_id FK
        string titre
        string categorie
        string site
        string source
        string chemin_fichier
        datetime date_upload
    }

    EMAIL_GENERE {
        int id PK
        int etudiant_id FK
        string type_email
        string destinataire
        string objet
        string contenu
        datetime date_creation
    }
```

In addition to the operational tables above, the `rag_chunks` table within PostgreSQL stores chunk texts alongside a 384-dimensional `vector(384)` column enabled by the `pgvector` extension.

---

## Prerequisites & System Requirements

- **Operating System**: Windows 10/11, Ubuntu 22.04+, or macOS.
- **Node.js**: `v18.x` or `v20.x` LTS.
- **Python**: `3.10` to `3.12`.
- **Database Services**:
  - PostgreSQL 16 with `pgvector` extension.
  - Neo4j Community or Enterprise Edition (v5+).
- **LLM Runner**: [Ollama](https://ollama.com/) (installed locally with model `qwen2.5:7b`) or a valid [Groq Cloud API Key](https://console.groq.com/).
- **Recommended Hardware for Local Inference**:
  - NVIDIA GPU with $\ge 8\text{ GB}$ VRAM (e.g., RTX 3060 / 4060 / 5070).
  - 16 GB to 32 GB System RAM.

---

## Installation & Quick Start

### Option A: Docker Compose (Recommended)

The easiest way to launch the entire platform with all dependencies:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PFA
   ```

2. **Configure environment**:
   ```bash
   cp .env.docker .env
   # Edit .env to set your passwords or GROQ_API_KEY if desired
   ```

3. **Start all services**:
   ```bash
   docker compose up --build -d
   ```

4. **Access the platform**:
   - **Frontend App**: `http://localhost`
   - **Backend API**: `http://localhost:8000/docs`
   - **RAG Microservice**: `http://localhost:8002/docs`
   - **Neo4j Browser**: `http://localhost:7474` (User: `neo4j`, Password: `password123`)

---

### Option B: Local Development Scripts (Windows)

For active local development without containerization:

1. **Set up Python Virtual Environments**:
   ```powershell
   # Backend virtualenv
   cd backend
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   deactivate
   cd ..

   # RAG Core virtualenv
   cd "RAG Core"
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   deactivate
   cd ..
   ```

2. **Set up Frontend Dependencies**:
   ```powershell
   cd frontend
   npm install
   cd ..
   ```

3. **Ensure Local Services Are Running**:
   - PostgreSQL (port `5432` or `5433` with pgvector installed).
   - Neo4j (bolt on `7687`, HTTP on `7474`).
   - Ollama with model downloaded: `ollama pull qwen2.5:7b`.

4. **Launch All Services at Once**:
   Double click or execute the batch script:
   ```cmd
   start_services.bat
   ```
   This script performs prerequisite checks, initiates PostgreSQL & Neo4j connectivity tests, starts Ollama, spins up the RAG microservice, runs the backend, and serves the Angular frontend on `http://localhost:4200`.

5. **Stop All Services**:
   ```cmd
   stop_services.bat
   ```

---

## Environment Configuration

Key configuration parameters (stored in `.env` / `backend/.env`):

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:password1234@localhost:5432/graphrag` |
| `SECRET_KEY` | JWT secret signing key | `changez-moi-en-production` |
| `RAG_API_URL` | Internal URL for RAG microservice | `http://localhost:8002` |
| `NEO4J_URI` | Neo4j Bolt connection string | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j database user | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j database password | `password123` |
| `OLLAMA_URL` | Ollama service endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default model for local inference | `qwen2.5:7b` |
| `GROQ_API_KEY` | Optional API key for cloud inference acceleration | *None* |
| `GROQ_MODEL` | Model to use when Groq key is present | `llama-3.1-8b-instant` |
| `ADMIN_EMAIL` | Default administrator account email | `admin@unihelp.tn` |
| `ADMIN_PASSWORD` | Default administrator initial password | `admin123` |

---

## API Endpoints Reference

### Backend Gateway (`http://localhost:8000`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/auth/login` | Authenticate user & issue JWT | No |
| `POST` | `/auth/register` | Student registration | No |
| `GET` | `/auth/me` | Fetch active user identity | Yes |
| `POST` | `/chat/query` | Send question to RAG, stream or return response | Yes |
| `GET` | `/conversations` | List user conversation threads | Yes |
| `POST` | `/conversations` | Initialize new chat conversation | Yes |
| `GET` | `/conversations/{id}/messages` | Retrieve full message thread history | Yes |
| `GET` | `/documents` | List uploaded institution documents | Admin |
| `POST` | `/documents/upload` | Upload new PDF/document & trigger ingestion | Admin |
| `DELETE`| `/documents/{id}` | Remove document and its associated chunks | Admin |
| `POST` | `/emails/generer` | Generate structured administrative email | Yes |
| `GET` | `/dashboard/stats` | Aggregate dashboard metrics and counts | Admin |

### RAG Microservice (`http://localhost:8002`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Healthcheck and model loading status |
| `POST` | `/process` | Parse, clean, and chunk uploaded files |
| `POST` | `/index-unindexed` | Embed unindexed chunks, sync Neo4j graph & BM25 |
| `POST` | `/query` | Execute hybrid search + cross-encoder rerank + generation |
| `POST` | `/generate-email` | Invoke LLM with administrative prompt templates |
| `DELETE`| `/chunks` | Purge chunks by institution and title |

---

## Testing & Evaluation

The repository includes a dedicated evaluation harness in the `tests/` directory:

- **Benchmark RAG Retrieval & Latency**:
  ```bash
  python tests/test_rag_performance.py
  ```
- **Verify Cross-Encoder Reranking Logic**:
  ```bash
  python tests/debug_rerank.py
  ```
- **End-to-End API Integration Suite**:
  ```bash
  python tests/test_api_full.py
  ```

---

## License

This project is developed as part of an Engineering Academic Final Year Project (Projet de Fin d'Année - PFA) for **IIT** and **ISB**. All rights reserved.

