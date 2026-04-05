import os
import json
import time
import random
import requests
import argparse
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
OUTPUT_FILE = "synthetic_chunks_pro.json"

# --- CREATIVE PROMPTS ---
GEN_PROMPT_STUDY_PLAN = """
Tu es un concepteur de programmes académiques pour l'IIT ou l'ISB (Tunisie).
Basé sur ce "seed" (exemple réel), imagine un NOUVEAU plan d'étude réaliste (autre année, autre spécialité).

RÈGLES :
1. Utilise le même style : codes de modules (ex: GM3-1, GC2 101), semestres (S1-S6), et crédits.
2. Invente des matières cohérentes (ex: IA Avancée, Robotique Marine, BIM 4D).
3. Garde le ton institutionnel francophone.

Retourne uniquement un JSON strict :
{{
  "page_content": "[SITE | TITRE] TEXTE_DU_PLAN_D_ETUDE",
  "metadata": {{
    "site": "IIT (ou ISB)",
    "title": "Titre inventé",
    "category": "wp_content",
    "content_type": "study_plan",
    "questions": ["3-5 questions types"],
    "keywords": ["10-15 mots clés"],
    "summary": "Bref résumé"
  }}
}}

SEED REEL :
{seed}
"""

GEN_PROMPT_MOBILITY = """
Tu es un responsable de la mobilité internationale à l'IIT/ISB Tunisie.
Basé sur ce "seed", imagine une NOUVELLE opportunité de partenariat ou une procédure de visa.

RÈGLES :
1. Invente une nouvelle université partenaire (Europe, Canada, USA) ou une nouvelle règle de bourse.
2. Détaille les coûts, les dates limites et les critère d'éligibilité.

Retourne uniquement un JSON strict :
{{
  "page_content": "[SITE | MOBILITE] TEXTE_DU_PARTENARIAT",
  "metadata": {{
    "site": "IIT (ou ISB)",
    "title": "Titre inventé",
    "category": "international",
    "content_type": "mobility",
    "questions": ["3-5 questions types"],
    "keywords": ["10-15 mots clés"],
    "summary": "Bref résumé"
  }}
}}

SEED REEL :
{seed}
"""

GEN_PROMPT_GENERAL = """
Tu es un rédacteur pour le site web de l'IIT/ISB Tunisie.
Imagine un NOUVEL article, une annonce, ou une procédure administrative réaliste.

RÈGLES :
1. Contenu : Vie étudiante, inscriptions, événements, certifications (DELF, IELTS, SAP).
2. Ton : Officiel et accueillant.

Retourne uniquement un JSON strict :
{{
  "page_content": "[SITE | ANNONCE] TEXTE_DE_L_ANNONCE",
  "metadata": {{
    "site": "IIT (ou ISB)",
    "title": "Titre inventé",
    "category": "general",
    "content_type": "general",
    "questions": ["3-5 questions types"],
    "keywords": ["10-15 mots clés"],
    "summary": "Bref résumé"
  }}
}}

SEED REEL :
{seed}
"""

class SyntheticGenerator:
    def __init__(self, seed_files):
        self.seeds = []
        for file in seed_files:
            if os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    self.seeds.extend(json.load(f))
        
        if not self.seeds:
            raise ValueError("No seed data found. Please run augmentation first.")

    def call_ollama(self, prompt):
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.8} # High temperature for creativity
            }, timeout=90)
            
            raw = response.json().get('response', '{}').strip()
            return json.loads(raw)
        except Exception as e:
            return None

    def generate_one(self):
        seed = random.choice(self.seeds)
        content_type = seed.get('metadata', {}).get('content_type', 'general')
        
        if content_type == "study_plan":
            prompt = GEN_PROMPT_STUDY_PLAN.format(seed=seed['page_content'])
        elif content_type == "mobility":
            prompt = GEN_PROMPT_MOBILITY.format(seed=seed['page_content'])
        else:
            prompt = GEN_PROMPT_GENERAL.format(seed=seed['page_content'])
            
        synthetic = self.call_ollama(prompt)
        if not synthetic or "page_content" not in synthetic:
            return None
            
        # Ensure enriched_content is present
        questions = " ".join(synthetic['metadata'].get('questions', []))
        summary = synthetic['metadata'].get('summary', "")
        keywords = " ".join(synthetic['metadata'].get('keywords', []))
        
        synthetic["enriched_content"] = f"{summary} {questions} {keywords} {synthetic['page_content']}"
        synthetic["metadata"]["augmentation"] = "synthetic_generation"
        
        return synthetic

    def run(self, count):
        results = []
        # Resume if file exists
        if os.path.exists(OUTPUT_FILE):
             with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                 results = json.load(f)
             print(f"🔄 Resuming... {len(results)} already generated.")

        needed = count - len(results)
        if needed <= 0:
            print("✅ Already reached the target count.")
            return

        print(f"🚀 Generating {needed} synthetic chunks (Target: {count})...")
        for i in tqdm(range(needed)):
            item = self.generate_one()
            if item:
                results.append(item)
            
            # Save checkpoint every 10 items
            if (i+1) % 10 == 0:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        # Final save
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"🎉 Success! Generated {len(results)} chunks saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="Total synthetic chunks to generate")
    args = parser.parse_args()

    # We use our 'augmented' chunks as seeds because they already have good labels
    seeds = ['iit_chunks_augmented.json', 'isb_chunks_augmented.json']
    
    gen = SyntheticGenerator(seeds)
    gen.run(args.count)
