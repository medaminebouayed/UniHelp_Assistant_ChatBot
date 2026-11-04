"""
retriever — Unified search combining dense, sparse, and graph retrieval.

Returns deduplicated results with weighted RRF fusion.
"""

from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import re

from rag.dense_index import dense_search
from rag.sparse_index import sparse_search
from rag.config import RRF_K, DENSE_WEIGHT, SPARSE_WEIGHT, TOP_K_DENSE, TOP_K_SPARSE

_neo4j_driver = None

def _get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import GraphDatabase
        from rag.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _neo4j_driver


# French morphological variants for query expansion
_FR_ACCENT_MAP = {
    "e": "éèêë", "a": "àâä", "i": "îï", "o": "ôö", "u": "ùûü", "c": "ç",
}


_FR_EXPANSIONS = {
    "accrédité": ["accrédité", "accréditée", "accrédités", "accréditées",
                  "accréditation", "accréditations", "accrédite", "accréditent",
                  "accréditer", "accrédité",
                  "accredite", "accredites", "accreditation", "accreditations",
                  "acrédité", "acréditée", "acrédités", "acréditées",
                  "acréditation", "acréditations",
                  "accredite", "accredites", "accreditations", "accreditation",
                  "accreditee", "accreditees"],
    "partenaire": ["partenaire", "partenaires", "partenariat", "partenariats",
                   "partenarial", "partenariaux"],
    "bourse": ["bourse", "bourses", "boursier", "boursière", "boursiers",
               "boursières", "boursier"],
    "frais": ["frais", "frais", "frais_scolarite", "frais_scolarité",
              "frais_inscription", "frais_universitaire", "frais_universitaires",
              "tarif", "tarifs", "cout", "couts", "coute"],
    "inscription": ["inscription", "inscriptions", "inscrire", "inscrit",
                    "inscrite", "inscrits", "inscrites",
                    "admission", "admissions", "candidature", "candidatures",
                    "postuler", "postule", "postulent", "sinscrire"],
    "laboratoire": ["laboratoire", "laboratoires", "labo", "labos"],
    "recherche": ["recherche", "recherches", "rechercher", "recherché",
                  "recherchée", "recherchés", "recherchées"],
    "formation": ["formation", "formations", "former", "formé", "formée",
                  "formés", "formées", "formant",
                  "cours", "module", "modules", "matiere", "matieres",
                  "enseignement", "enseignements", "cursus", "programme",
                  "programmes", "filiere", "filieres", "specialite",
                  "specialites", "spécialité", "spécialités"],
    "stage": ["stage", "stages", "stagiaire", "stagiaires",
              "alternance", "alternances", "apprentissage", "professionnel",
              "professionnelle"],
    "diplôme": ["diplôme", "diplômes", "diplômé", "diplômée", "diplômés",
                "diplômées", "diplomé", "diplomés", "diplomées",
                "diplome", "diplomes",
                "certificat", "certificats", "license", "licence",
                "master", "doctorat", "doctorats"],
    "université": ["université", "universités", "universitaire",
                   "universitaires", "universite", "universites",
                   "ecole", "institut", "faculte", "faculté",
                   "facultes", "facultés", "etablissement",
                   "etablissements"],
    "étudiant": ["étudiant", "étudiante", "étudiants", "étudiantes",
                 "etudiant", "etudiante", "etudiants", "etudiantes",
                 "étudier", "etudier",
                 "eleve", "eleves", "apprenant", "apprenants"],
    "mobilité": ["mobilité", "mobilités", "mobilite", "mobilites",
                 "echange", "echanges", "échange", "échanges",
                 "international", "internationaux", "internationale",
                 "internationales", "erasmus", "programme_echange",
                 "semestre_etranger", "semestre_abroad",
                 "etranger", "etrangers", "étranger", "étrangers"],
}


def _accent_fold(w: str) -> str:
    """Remplace les caractères accentués par leur version sans accent."""
    replacements = [
        ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
        ("à", "a"), ("â", "a"), ("ä", "a"),
        ("î", "i"), ("ï", "i"),
        ("ô", "o"), ("ö", "o"),
        ("ù", "u"), ("û", "u"), ("ü", "u"),
        ("ç", "c"),
        ("É", "E"), ("È", "E"), ("Ê", "E"), ("Ë", "E"),
        ("À", "A"), ("Â", "A"), ("Ä", "A"),
        ("Î", "I"), ("Ï", "I"),
        ("Ô", "O"), ("Ö", "O"),
        ("Ù", "U"), ("Û", "U"), ("Ü", "U"),
        ("Ç", "C"),
    ]
    for a, b in replacements:
        w = w.replace(a, b)
    return w

def _expand_query(query: str) -> str:
    """Génère des variantes morphologiques françaises pour la recherche BM25."""
    q_lower = query.lower()
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:-[A-Za-zÀ-ÿ0-9]+)*", q_lower)
    token_folded = {_accent_fold(t) for t in tokens}
    added = []
    for tok in tokens:
        if len(tok) < 3:
            continue
        tok_folded = _accent_fold(tok)
        for canonical, variants in _FR_EXPANSIONS.items():
            all_forms = [canonical] + variants
            all_folded = [_accent_fold(f) for f in all_forms]
            if tok in all_forms or tok_folded in all_folded:
                for form in all_forms:
                    ff = _accent_fold(form)
                    if ff not in token_folded and ff not in [_accent_fold(a) for a in added]:
                        added.append(form)
                break
    if not added:
        return query
    return query + " " + " ".join(added)


def _dedup(results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Keep best score per doc_key."""
    best: Dict[str, float] = {}
    for doc, score in results:
        if doc not in best or score > best[doc]:
            best[doc] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _rrf(
    dense: List[Tuple[str, float]],
    sparse: List[Tuple[str, float]],
    graph: Optional[List[Tuple[str, float]]] = None,
    k: int = RRF_K,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    graph_weight: float = 0.20,
) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for rank, (doc, _) in enumerate(dense, start=1):
        scores[doc] += dense_weight / (k + rank)
    for rank, (doc, _) in enumerate(sparse, start=1):
        scores[doc] += sparse_weight / (k + rank)
    if graph:
        for rank, (doc, _) in enumerate(graph, start=1):
            scores[doc] += graph_weight / (k + rank)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


def hybrid_search(
    query: str,
    top_k_dense: int = TOP_K_DENSE,
    top_k_sparse: int = TOP_K_SPARSE,
    use_graph: bool = False,
) -> List[str]:
    """Search all indexes → RRF fusion → ranked doc_keys."""
    d_res = _dedup(dense_search(query, k=top_k_dense))
    expanded = _expand_query(query)
    s_res = _dedup(sparse_search(expanded, k=top_k_sparse))
    g_res = None
    if use_graph:
        try:
            from rag.graph_index import _extract_entities
            driver = _get_neo4j_driver()
            entities = _extract_entities(query)
            doc_scores: Dict[str, float] = defaultdict(float)
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
                        site = (record["site"] or "").strip().lower()
                        sec = (record["section_title"] or "").strip().lower()
                        dk = f"{site}|{sec}"
                        doc_scores[dk] += 1.0
            g_res = _dedup(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100])
            if g_res:
                print(f"[retriever] Graph search: {len(g_res)} results from {len(entities)} terms")
        except Exception as e:
            print(f"[retriever] Graph search failed: {e}")
    return _rrf(d_res, s_res, g_res)
