"""
augmentation_script.py — Full corrected version
=================================================
Fixes over previous version:
  FIX 1 — section_title: accepts multiple key names the LLM may use
  FIX 2 — header field is updated with the clean reconstructed title
  FIX 3 — bm25_text always written, includes section_title
  FIX 4 — section_title included in enriched_content (first position)
  FIX 5 — prompts explicitly enforce key names to avoid LLM renaming

Dependencies:
    pip install requests tqdm python-dotenv
"""

import os
import json
import time
import re
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


# ════════════════════════════════════════════════════════════════════════
# PROMPTS  (FIX 5 — explicit key name enforcement)
# ════════════════════════════════════════════════════════════════════════

PROMPT_STUDY_PLAN = """
Tu es un expert en contenu académique francophone (IIT/ISB Tunisie).
Cet extrait est un plan d'études contenant des codes de modules, semestres et spécialisations.

Génère les données d'augmentation au format JSON strict.
Retourne UNIQUEMENT le JSON valide. Aucun texte avant ou après.
Utilise EXACTEMENT ces noms de clés: questions, keywords, section_title, summary

{{
  "questions": [
    "4 questions précises qu'un étudiant poserait sur ce plan d'étude (en français)"
  ],
  "keywords": [
    "tous les codes de modules (ex: GC1 101), noms de matières, numéros de semestres (S1/S2/S3), groupes de modules (GM1-1), spécialisations, niveaux (Bac+3) — 10 à 15 termes"
  ],
  "section_title": "titre de section complet reconstruit (ex: Plan d'études Génie Civil 1ère année)",
  "summary": "1-2 phrases résumant ce plan d'études"
}}

TEXTE:
{chunk}
"""

PROMPT_MOBILITY = """
Tu es un expert en mobilité internationale étudiante (IIT/ISB Tunisie vers France/Europe).
Cet extrait décrit une université partenaire, un programme d'échange ou une procédure de mobilité.

Génère les données d'augmentation au format JSON strict.
Retourne UNIQUEMENT le JSON valide. Aucun texte avant ou après.
Utilise EXACTEMENT ces noms de clés: questions, keywords, section_title, summary

{{
  "questions": [
    "4-5 questions qu'un étudiant tunisien poserait sur ce partenariat ou cette mobilité (en français)"
  ],
  "keywords": [
    "noms d'universités partenaires, villes, pays, coûts en euros (ex: 670€), durées de stage, diplômes, dates limites, accréditations — 10 à 15 termes"
  ],
  "section_title": "titre de section complet reconstruit (ex: Partenariat 3iL Ingénieurs — Mobilité France)",
  "summary": "1-2 phrases résumant ce partenariat ou programme de mobilité"
}}

TEXTE:
{chunk}
"""

PROMPT_GENERAL = """
Tu es un expert en contenu institutionnel éducatif francophone (IIT/ISB Tunisie).
Cet extrait provient du site web ou d'un document institutionnel.

Génère les données d'augmentation au format JSON strict.
Retourne UNIQUEMENT le JSON valide. Aucun texte avant ou après.
Utilise EXACTEMENT ces noms de clés: questions, keywords, section_title, summary

{{
  "questions": [
    "3-4 questions qu'un étudiant ou parent poserait sur ce contenu (en français)"
  ],
  "keywords": [
    "noms de programmes, accréditations, compétences, chiffres importants, acronymes, noms propres — 8 à 12 termes"
  ],
  "section_title": "titre de section complet reconstruit",
  "summary": "1-2 phrases résumant le contenu"
}}

TEXTE:
{chunk}
"""


# ════════════════════════════════════════════════════════════════════════
# CONTENT TYPE DETECTOR
# ════════════════════════════════════════════════════════════════════════

def detect_content_type(chunk: dict) -> str:
    category = chunk["metadata"].get("category", "").lower()
    content  = chunk["page_content"].lower()
    title    = chunk["metadata"].get("title", "").lower()

    has_module_codes = bool(re.search(r'\b[A-Z]{2}\d[-\s]\d{3}\b', chunk["page_content"]))
    has_sem_labels   = any(x in content for x in ["s1 ", "s2 ", "s3 ", "s4 ", "semestre"])
    has_plan_title   = any(x in title   for x in ["plan", "etude", "programme"])

    if category == "wp_content" or has_module_codes or (has_sem_labels and has_plan_title):
        return "study_plan"

    if any(cat in category for cat in ["mobilite_internationale", "international"]):
        return "mobility"

    return "general"


# ════════════════════════════════════════════════════════════════════════
# OLLAMA CALL
# ════════════════════════════════════════════════════════════════════════

def call_ollama(prompt: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model"  : OLLAMA_MODEL,
                    "prompt" : prompt,
                    "stream" : False,
                    "format" : "json",
                    "options": {"temperature": 0, "num_predict": 700},
                },
                timeout=90,
            )
            raw = response.json().get("response", "{}").strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'\s*```$',          '', raw, flags=re.MULTILINE)
            return json.loads(raw.strip())

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    [JSON error] {e}")
                return {}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    [Ollama error] {e}")
                return {}
    return {}


# ════════════════════════════════════════════════════════════════════════
# SECTION TITLE EXTRACTOR  (FIX 1)
# ════════════════════════════════════════════════════════════════════════

def extract_section_title(aug: dict) -> str:
    """FIX 1 — accept any key name the LLM might use for the section title."""
    for key in ["section_title", "titre_section", "titre", "title", "section", "heading"]:
        val = aug.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return ""


# ════════════════════════════════════════════════════════════════════════
# AUGMENT ONE CHUNK
# ════════════════════════════════════════════════════════════════════════

def augment_chunk(chunk: dict) -> dict:
    content_type = detect_content_type(chunk)
    text = chunk["page_content"]

    prompt_map = {
        "study_plan": PROMPT_STUDY_PLAN,
        "mobility"  : PROMPT_MOBILITY,
        "general"   : PROMPT_GENERAL,
    }
    aug = call_ollama(prompt_map[content_type].format(chunk=text))

    questions     = [q for q in aug.get("questions", []) if isinstance(q, str)]
    keywords      = [k for k in aug.get("keywords",  []) if isinstance(k, str)]
    summary       = aug.get("summary", "")
    section_title = extract_section_title(aug)          # FIX 1

    # ── FIX 2: replace broken truncated header with clean title ─────────
    if section_title:
        chunk["metadata"]["header_original"] = chunk["metadata"].get("header", "")
        chunk["metadata"]["header"]          = section_title

    # ── Store augmentation in metadata ───────────────────────────────────
    chunk["metadata"]["content_type"]  = content_type
    chunk["metadata"]["questions"]     = questions
    chunk["metadata"]["keywords"]      = keywords
    chunk["metadata"]["section_title"] = section_title
    chunk["metadata"]["summary"]       = summary

    # ── FIX 3 & 4: enriched_content — section_title first ───────────────
    chunk["enriched_content"] = " ".join(filter(None, [
        section_title,                  # FIX 4: clean title first
        summary,
        " ".join(questions),
        " ".join(keywords),
        text,
    ]))

    # ── FIX 3: bm25_text — always written, no breadcrumb ────────────────
    chunk["bm25_text"] = " ".join(filter(None, [
        section_title,
        summary,
        " ".join(keywords),
    ]))

    return chunk


# ════════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_augmentation(
    input_file : str,
    output_file: str,
    limit      : int = None,
    start_from : int = 0,
    batch_size : int = 20,
):
    if not os.path.exists(input_file):
        print(f"[warn] {input_file} not found.")
        return

    with open(input_file, encoding="utf-8") as f:
        chunks = json.load(f)

    to_process = chunks[:limit] if limit else chunks
    total      = len(to_process)

    output_path = Path(output_file)
    if output_path.exists() and start_from > 0:
        with open(output_path, encoding="utf-8") as f:
            augmented = json.load(f)
        print(f"Resuming from chunk {start_from} ({len(augmented)} already done).")
    else:
        augmented = []

    type_counts = {"study_plan": 0, "mobility": 0, "general": 0}
    empty_aug   = 0

    print(f"\nAugmenting {total - start_from} chunks from {input_file}...")

    for i in tqdm(range(start_from, total)):
        chunk = to_process[i].copy()

        if len(chunk["page_content"]) < 30:
            chunk["metadata"].update({
                "content_type": "noise", "questions": [],
                "keywords": [], "section_title": "", "summary": "",
            })
            chunk["enriched_content"] = chunk["page_content"]
            chunk["bm25_text"]        = ""
            augmented.append(chunk)
            continue

        chunk = augment_chunk(chunk)
        ct    = chunk["metadata"]["content_type"]
        if ct in type_counts:
            type_counts[ct] += 1
        if not chunk["metadata"].get("summary"):
            empty_aug += 1
        augmented.append(chunk)

        if (i + 1) % batch_size == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(augmented, f, ensure_ascii=False, indent=2)
            tqdm.write(f"  Checkpoint: {len(augmented)}/{total}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print(f"\nDone — {len(augmented)} chunks saved to {output_file}")
    print(f"Content types : {type_counts}")
    print(f"Empty aug     : {empty_aug}")


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site",       choices=["iit", "isb", "both"], default="both")
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    args = parser.parse_args()

    # Paths updated for reorganized structure
    if args.site in ("iit", "both"):
        run_augmentation(
            "data/raw/iit_chunks.json", "data/augmented/iit_chunks_augmented.json",
            limit=args.limit, start_from=args.start_from,
        )
    if args.site in ("isb", "both"):
        run_augmentation(
            "data/raw/isb_chunks.json", "data/augmented/isb_chunks_augmented.json",
            limit=args.limit,
            start_from=args.start_from if args.site == "isb" else 0,
        )