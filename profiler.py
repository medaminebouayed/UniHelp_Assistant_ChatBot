import time
import asyncio
import sys
import os
sys.path.insert(0, r"c:\Users\amine\OneDrive\Desktop\SoftwareEngenering\ING2\PFA\RAG Core")

from rag.pipeline import query
from rag_api.main import _ask_llm, _build_prompt

async def profile_rag():
    question = "donner moi le plan d'étude de génie informatique"
    print(f"--- Profiling Question: {question} ---")
    
    t0 = time.time()
    candidates = query(question, top_k=10, use_graph=True)
    t1 = time.time()
    print(f"Retrieval + Reranking: {t1 - t0:.2f} seconds")
    
    # Let's break down retrieval manually
    from rag.retriever import dense_search, sparse_search, _expand_query
    
    t_d0 = time.time()
    dense_search(question, k=40)
    t_d1 = time.time()
    print(f"  -> Dense Search: {t_d1 - t_d0:.2f} seconds")
    
    t_s0 = time.time()
    expanded = _expand_query(question)
    sparse_search(expanded, k=60)
    t_s1 = time.time()
    print(f"  -> Sparse Search: {t_s1 - t_s0:.2f} seconds")
    
    t_g0 = time.time()
    from rag.graph_index import _extract_entities
    from neo4j import GraphDatabase
    from rag.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    entities = _extract_entities(question)
    doc_scores = {}
    with driver.session() as session:
        terms = entities[:10]
        if terms:
            result = session.run("""
                UNWIND $terms AS term
                MATCH (n)
                WHERE NOT n:Chunk
                  AND (
                    toLower(toString(n.name)) CONTAINS toLower(term)
                    OR toLower(toString(n.canonical_name)) CONTAINS toLower(term)
                  )
                MATCH (n)-[:MENTIONED_IN]-(ch:Chunk)
                RETURN ch.site AS site, ch.section_title AS section_title
                LIMIT 300
            """, terms=terms)
            for record in result:
                pass
    driver.close()
    t_g1 = time.time()
    print(f"  -> Graph Search: {t_g1 - t_g0:.2f} seconds")
    
    from rag.reranker import rerank
    t_r0 = time.time()
    pool = ["IIT|Admissions"] * 50 # dummy
    rerank(question, pool)
    t_r1 = time.time()
    print(f"  -> Reranker (50 items): {t_r1 - t_r0:.2f} seconds")

    context = "Context dummy"
    t_l0 = time.time()
    answer = await _ask_llm(question, context)
    t_l1 = time.time()
    print(f"LLM Generation (Ollama): {t_l1 - t_l0:.2f} seconds")
    
    print(f"Total Time: {t1 - t0 + t_l1 - t_l0:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(profile_rag())
