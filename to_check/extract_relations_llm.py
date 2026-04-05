"""
extract_relations_llm.py — Extraction de relations avec Qwen 2.5 (Ollama)
=========================================================================
Ce script :
    1. Lit les chunks depuis Neo4j
    2. Envoie chaque chunk à Qwen 2.5
    3. Qwen détecte les entités et relations
    4. Python crée les nouvelles relations dans Neo4j

Relations détectées par LLM :
    - A_FRAIS      : Programme → Frais
    - ADMET_VIA    : Programme → Admission
    - A_OBJECTIF   : Programme → Objectif
    - SITUE_A      : Institution → Localisation
    - OFFRE_METIER : Programme → Metier

Usage:
    1. Assure-toi qu'Ollama tourne : ollama run qwen2.5
    2. python extract_relations_llm.py
"""

import json
import requests
from neo4j import GraphDatabase

# ── CONFIG ─────────────────────────────────────────────────────────────
NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"   # ← Ton mot de passe

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "qwen2.5"

# Nombre de chunks à traiter (commence petit pour tester)
LIMIT_CHUNKS   = 20


# ── Prompt envoyé à Qwen ───────────────────────────────────────────────
EXTRACTION_PROMPT = """
Tu es un expert en extraction d'informations pour des graphes de connaissances.

Analyse ce texte extrait d'un document universitaire tunisien et extrais les entités et relations.

TEXTE :
{content}

INSTRUCTIONS :
- Extrais UNIQUEMENT les informations présentes dans le texte
- Ne devine pas, n'invente pas
- Réponds UNIQUEMENT en JSON valide, rien d'autre

FORMAT JSON attendu :
{{
  "entites": [
    {{"label": "Frais",       "valeur": "montant en DT si mentionné"}},
    {{"label": "Admission",   "valeur": "conditions d'admission si mentionnées"}},
    {{"label": "Objectif",    "valeur": "objectif du programme si mentionné"}},
    {{"label": "Metier",      "valeur": "débouchés professionnels si mentionnés"}},
    {{"label": "Localisation","valeur": "ville ou adresse si mentionnée"}}
  ],
  "relations": [
    {{"type": "A_FRAIS",      "valeur": "montant si trouvé"}},
    {{"type": "ADMET_VIA",    "valeur": "condition si trouvée"}},
    {{"type": "A_OBJECTIF",   "valeur": "objectif si trouvé"}},
    {{"type": "OFFRE_METIER", "valeur": "métier si trouvé"}},
    {{"type": "SITUE_A",      "valeur": "lieu si trouvé"}}
  ]
}}

Si une information n'est pas dans le texte, ne l'inclus pas dans la liste.
"""


# ── Appel à Ollama ─────────────────────────────────────────────────────
def call_qwen(content: str) -> dict | None:
    """Envoie le chunk à Qwen et retourne le JSON extrait."""
    prompt = EXTRACTION_PROMPT.format(content=content[:800])  # limite 800 chars

    try:
        response = requests.post(OLLAMA_URL, json={
            "model" : OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}  # 0 = réponses déterministes
        }, timeout=60)

        raw = response.json()["response"].strip()

        # Nettoyer les backticks si Qwen les ajoute
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"    ⚠️ JSON invalide : {e}")
        return None
    except Exception as e:
        print(f"    ⚠️ Erreur Ollama : {e}")
        return None


# ── Créer les relations dans Neo4j ─────────────────────────────────────
def create_llm_relations(tx, chunk_id: str, programme: str, site: str, extracted: dict):
    """
    Pour chaque relation détectée par le LLM :
    Crée un nouveau nœud et le relie au Programme.

    Exemple :
        (Programme: Génie Civil) -[:A_FRAIS]-> (Frais: {valeur: "3500 DT"})
    """
    relations = extracted.get("relations", [])

    for rel in relations:
        rel_type = rel.get("type", "").strip()
        valeur   = rel.get("valeur", "").strip()

        if not rel_type or not valeur or valeur.lower() in ["non mentionné", "n/a", ""]:
            continue

        # Mapping relation → label du nœud cible
        label_map = {
            "A_FRAIS"     : "Frais",
            "ADMET_VIA"   : "Admission",
            "A_OBJECTIF"  : "Objectif",
            "OFFRE_METIER": "Metier",
            "SITUE_A"     : "Localisation",
        }

        node_label = label_map.get(rel_type, "Info")

        # Crée le nœud cible + la relation vers le Programme
        query = f"""
            MATCH (p:Programme {{name: $programme, site: $site}})
            MERGE (n:{node_label} {{valeur: $valeur}})
            MERGE (p)-[:{rel_type}]->(n)
        """
        tx.run(query, {
            "programme": programme,
            "site"     : site,
            "valeur"   : valeur,
        })


# ── Main ───────────────────────────────────────────────────────────────
def main():
    # Connexion Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("✅ Connecté à Neo4j\n")

    # Récupérer les chunks depuis Neo4j
    print(f"📂 Récupération de {LIMIT_CHUNKS} chunks depuis Neo4j...")
    with driver.session() as session:
        result = session.run("""
            MATCH (i:Institution)-[:OFFRE]->(p:Programme)-[:CONTIENT]->(c:Chunk)
            RETURN
                c.id      AS chunk_id,
                c.content AS content,
                p.name    AS programme,
                i.name    AS site
            LIMIT $limit
        """, {"limit": LIMIT_CHUNKS})
        chunks = [dict(r) for r in result]

    print(f"   → {len(chunks)} chunks récupérés\n")

    # Traitement chunk par chunk
    success = 0
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] Programme: {chunk['programme'][:40]}")

        # Appel LLM
        extracted = call_qwen(chunk["content"])

        if not extracted:
            print("    ⏭️ Ignoré (pas de JSON valide)")
            continue

        # Afficher ce que Qwen a trouvé
        relations = extracted.get("relations", [])
        found = [r["type"] for r in relations if r.get("valeur", "")]
        if found:
            print(f"    ✅ Relations trouvées : {', '.join(found)}")
        else:
            print(f"    ℹ️ Aucune relation détectée dans ce chunk")

        # Créer dans Neo4j
        with driver.session() as session:
            session.execute_write(
                create_llm_relations,
                chunk["chunk_id"],
                chunk["programme"],
                chunk["site"],
                extracted
            )
        success += 1

    driver.close()

    print(f"\n🎉 Terminé ! {success}/{len(chunks)} chunks traités.")
    print("→ Dans Neo4j Browser, tape :")
    print("  MATCH (p:Programme)-[r]->(n) RETURN p,r,n LIMIT 50")


if __name__ == "__main__":
    main()
