# Graph RAG Implementation Plan

Building a robust Graph RAG system for IIT/ISB involves moving beyond simple semantic search to structured knowledge retrieval. This allows the system to answer complex, multi-hop questions (e.g., "What are the registration steps for a student coming from a Swiss university seeking a Master's in Civil Engineering?").

## Phase 1: Schema Design & Node Definition

### 1. Entities (Nodes)
- **Institution**: `IIT` (International Institute of Technology), `ISB` (International School of Business).
- **Program**: Specific degrees like `Génie Civil`, `Génie Informatique`, `Licence Management`.
- **Level**: `Licence`, `Master`, `Cycle Ingénieur`, `MBA`.
- **Requirement/Document**: `Passport`, `Diploma`, `CVEC`, `VLS-TS Visa`, `English Certificate`.
- **Procedure**: `Inscription`, `Bourse`, `Visa Application`.
- **Contact/Entity**: `Cellule d'écoute`, `Direction des études`, `Service International`.
- **Chunk**: The raw text chunks from [iit_chunks.json](file:///c:/Users/amine/OneDrive/Desktop/SoftwareEngenering/ING2/PFA/iit_chunks.json) (linked to the entities they describe).

### 2. Relationships (Edges)
- [(Institution)-[:OFFERS]->(Program)](file:///c:/Users/amine/OneDrive/Desktop/SoftwareEngenering/ING2/PFA/utils/chunk_data.py#45-63)
- [(Program)-[:HAS_LEVEL]->(Level)](file:///c:/Users/amine/OneDrive/Desktop/SoftwareEngenering/ING2/PFA/utils/chunk_data.py#45-63)
- [(Program)-[:REQUIRES]->(Requirement)](file:///c:/Users/amine/OneDrive/Desktop/SoftwareEngenering/ING2/PFA/utils/chunk_data.py#45-63)
- [(Procedure)-[:INVOLVES_DOCUMENT]->(Requirement)](file:///c:/Users/amine/OneDrive/Desktop/SoftwareEngenering/ING2/PFA/utils/chunk_data.py#45-63)
- [(Procedure)-[:MANAGED_BY]->(Contact)](file:///c:/Users/amine/OneDrive/Desktop/SoftwareEngenering/ING2/PFA/utils/chunk_data.py#45-63)
- [(Chunk)-[:MENTIONS]->(AnyEntity)](file:///c:/Users/amine/OneDrive/Desktop/SoftwareEngenering/ING2/PFA/utils/chunk_data.py#45-63)
- `(Chunk)-[:PART_OF]->(Program/Procedure)`

---

## Phase 2: Extraction Pipeline (KGE - Knowledge Graph Extraction)

### 1. Extraction Script (`utils/extract_entities.py`)
- **Input**: `iit_chunks.json` & `isb_chunks.json`.
- **Logic**: Use **Ollama** with **Qwen 2.5 (7B)** or **Llama 3.1 (8B)**. Qwen 2.5 is recommended for its superior JSON structure adherence.
- **Output**: `knowledge_graph.json` containing normalized nodes and edges.

### 2. De-duplication & Merging
- Handle "Génie Civil" vs "Civil Engineering" entities by mapping them to standardized IDs.
- Merge identical nodes from different chunks (Entity Resolution).

---

## Phase 3: Graph Storage & Hybrid Retrieval

### 1. Neo4j Integration
- **Setup**: Use Neo4j AuraDB (Free Tier) or a local Docker container.
- **Ingestion**: A script (`utils/ingest_graph.py`) to run Cypher `MERGE` commands.

### 2. Hybrid Query Logic
- **Step A**: Retrieve top 3-5 snippets using Vector Search (ChromaDB).
- **Step B**: Identify the "Core Entities" in the snippets.
- **Step C**: Query Neo4j for all relationships connected to these entities (1-2 hops).
- **Step D**: Combine Vector Context + Graph Facts for the final LLM response.

---

## Phase 4: Implementation Task List

- [ ] **Step 1**: Create `utils/extract_entities.py` (The Extraction LLM Script).
- [ ] **Step 2**: Setup Neo4j instance and connectivity.
- [ ] **Step 3**: Run extraction on [iit_chunks.json](file:///c:/Users/amine/OneDrive/Desktop/SoftwareEngenering/ING2/PFA/iit_chunks.json).
- [ ] **Step 4**: Implement the `GraphRetrieval` class to bridge Neo4j and LangChain.
- [ ] **Step 5**: Update the main Chatbot to use the new Hybrid context.

## Verification Plan
1. **Fact Check**: Query "Which documents are needed for the Master's in IT?" and verify the list matches the graph connections perfectly.
2. **Multi-hop Test**: "Who manages the scholarship procedure for international students?" (Connects Procedure -> Contact).
