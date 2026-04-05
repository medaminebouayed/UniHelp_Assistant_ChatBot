import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load a lightweight model for comparison
device = "cuda" if torch.cuda.is_available() else "cpu"
# Using HF_TOKEN for authentication
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', token=HF_TOKEN).to(device)

def load_data(file):
    if not os.path.exists(file):
        print(f"⚠️ {file} not found.")
        return []
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_quality(data, name):
    if not data: return
    
    # Extract text
    texts = [str(item.get('page_content', '')) for item in data]
    
    # 1. Diversity (Unique Words)
    all_words = " ".join(texts).lower().split()
    if not all_words: return
    unique_words = len(set(all_words))
    diversity_score = unique_words / len(all_words)
    
    # 2. Semantic Similarity to itself (Redundancy check)
    # If similarity is too high (0.99+), it's just repetition
    embeddings = model.encode(texts[:50], convert_to_tensor=True) # Limit to 50 for speed
    cos_sim = util.cos_sim(embeddings, embeddings)
    avg_sim = torch.mean(cos_sim).item()
    
    print(f"\n--- Evaluation: {name} ---")
    print(f"📊 Total Samples: {len(texts)}")
    print(f"🔤 Diversity (Unique/Total Words): {diversity_score:.2f}")
    print(f"🔄 Redundancy (Self-Similarity): {avg_sim:.2f} (Lower = More varied)")
    print(f"📝 Avg Length: {pd.Series([len(t) for t in texts]).mean():.1f} chars")

def main():
    # Updated paths for reorganized structure
    ctgan_data = load_data('data/synthetic/synthetic_data_ctgan.json')
    llm_data = load_data('data/synthetic/synthetic_chunks_factual_pro.json')
    
    if ctgan_data:
        evaluate_quality(ctgan_data, "CTGAN (Tabular)")
    if llm_data:
        evaluate_quality(llm_data, "LLM (Semantic)")

if __name__ == "__main__":
    main()
