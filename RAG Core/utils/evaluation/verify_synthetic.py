import json
import os
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv

# Load for consistency across tools
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def extract_codes(text):
    """Extract sequences like GC1 101, GM2-1, etc."""
    return set(re.findall(r'\b[A-Z]{2}\d[-\s]\d{3}\b|\b[A-Z]{2}\d(?:-\d)?\b', text))

def check_semantic_recall(seed_file, synthetic_file, limit=10):
    if not os.path.exists(seed_file) or not os.path.exists(synthetic_file):
        print(f"⚠️ Fichier manquant: {seed_file} ou {synthetic_file}")
        return

    with open(seed_file, 'r', encoding='utf-8') as f:
        seeds = json.load(f)
    with open(synthetic_file, 'r', encoding='utf-8') as f:
        synths = json.load(f)

    print(f"\n🧪 --- DIAGNOSTIC DE FIDÉLITÉ (Top {limit}) ---")
    
    for i in range(min(limit, len(synths))):
        # Find matching seed (by cycling or by link)
        seed = seeds[i % len(seeds)]
        synth = synths[i]
        
        # 1. Linguistic Diversity (Lower is better for diversity)
        ratio = SequenceMatcher(None, seed['page_content'], synth['page_content']).ratio()
        
        # 2. Entity Recall (Must be 1.0)
        seed_codes = extract_codes(seed['page_content'])
        synth_codes = extract_codes(synth['page_content'])
        
        missing = seed_codes - synth_codes
        extra = synth_codes - seed_codes
        
        print(f"\n🔹 Chunk {i}: {seed['metadata'].get('title', 'N/A')}")
        print(f"   Linguistic Similarity: {ratio:.1%}")
        if missing:
             print(f"   ❌ Codes MANQUANTS: {missing}")
        elif extra:
             print(f"   ⚠️ Codes INVENTÉS: {extra}")
        else:
             print(f"   ✅ Codes Techniques 100% OK")
             
        if ratio > 0.90:
             print("   ⚠️ Attention: Trop proche de l'original (Peu de diversité)")
        elif ratio < 0.30:
             print("   ⚠️ Attention: Trop éloigné (Risque de perte de sens)")
        else:
             print("   🎯 Équilibre Sémantique Parfait")

if __name__ == "__main__":
    check_semantic_recall(
        'data/augmented/iit_chunks_augmented.json', 
        'data/synthetic/full_10k.json'
    )
