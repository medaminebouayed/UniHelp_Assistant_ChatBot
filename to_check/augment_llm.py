import json
import os
import sys
import random
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import warnings
import asyncio
from tqdm import tqdm
from collections import Counter

# Load environment variables from .env
load_dotenv()

# Configuration
GEMINI_MODEL = "gemini-1.5-flash-latest"
NUM_AUGMENTATIONS = 2
SAMPLE_SIZE = 2000

# Suppress warnings (Note: This hides warnings, but not errors)
warnings.filterwarnings('ignore')

async def augment_chunk(chunk_text, llm, retries=3):
    """Generate semantic paraphrases for a single chunk with retry logic"""
    
    # IMPROVED PROMPT: Strict rules to force diversity and remove filler
    prompt = ChatPromptTemplate.from_template("""
    You are a precise data augmentation engine.
    Generate {num} distinct rewrites of the following text.

    STRICT RULES:
    1. Maintain ALL facts, numbers, and entities exactly.
    2. DO NOT use introductory phrases (e.g., "The document states", "According to", "This section"). Start directly with the subject or fact.
    3. Keep the output length similar to the original. Do not add filler words.
    4. Use diverse vocabulary and sentence structures for each variation.

    Original:
    {text}

    Return exactly {num} variations, each starting with [VARIATION] on a new line.
    """)
    
    chain = prompt | llm
    
    for attempt in range(retries):
        try:
            response = await chain.ainvoke({"text": chunk_text, "num": NUM_AUGMENTATIONS})
            # Clean up the output
            variations = [v.strip().replace("[VARIATION]", "") for v in response.content.split("[VARIATION]") if v.strip()]
            return variations[:NUM_AUGMENTATIONS]
        except Exception as e:
            delay = 2 ** (attempt + 1) + random.uniform(0, 1)
            if "404" in str(e) or "NOT_FOUND" in str(e):
                 # Sometimes model name resolution fails temporarily
                 print(f"\n⚠️ Attempt {attempt+1} failed with 404. Retrying in {delay:.1f}s...")
            elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                 print(f"\n⏳ Rate limit hit. Retrying in {delay*2:.1f}s...")
                 delay *= 2 # Extra delay for rate limits
            else:
                 print(f"\n❌ Error augmenting chunk (Attempt {attempt+1}): {e}")
            
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                return []
    return []

def analyze_start_phrases(filename):
    """Checks if the LLM is using repetitive starting phrases"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        starts = []
        total_synthetic = 0
        
        for item in data:
            # Check if it's a synthetic row
            if item.get('metadata', {}).get('augmentation') == 'llm_paraphrase':
                total_synthetic += 1
                words = item.get('page_content', '').split()
                if len(words) >= 3:
                    starts.append(" ".join(words[:3]))

        if total_synthetic == 0:
            print("⚠️ Warning: No synthetic data was generated. Check API Key or Library Compatibility.")
            return

        common_starts = Counter(starts).most_common(5)
        
        print(f"\n🔍 Diagnostics on {total_synthetic} synthetic samples:")
        print("Top 5 starting phrases:")
        for phrase, count in common_starts:
            print(f"  '{phrase}...': {count} times ({(count/len(starts))*100:.1f}%)")
            
        top_count = common_starts[0][1]
        if (top_count / len(starts)) > 0.10:
            print("⚠️ Warning: High repetition of starting phrases detected.")
        else:
            print("✅ Good diversity in sentence structure.")
            
    except Exception as e:
        print(f"Could not run diagnostics: {e}")

async def main():
    # 1. Check Python Version Compatibility
    if sys.version_info >= (3, 14):
        print("⚠️ WARNING: You are running Python 3.14+.")
        print("   LangChain and Pydantic may not be stable on this version.")
        print("   Recommended: Use Python 3.10, 3.11, or 3.12.")
        print("-" * 50)

    # 2. Load API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("🛑 Error: GOOGLE_API_KEY not found in environment variables.")
        return

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)

    # 3. Load Data
    chunk_files = ['iit_chunks.json', 'isb_chunks.json']
    all_chunks = []
    for file in chunk_files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                all_chunks.extend(json.load(f))

    if not all_chunks:
        print("❌ No data loaded. Check if .json files exist.")
        return

    # 4. Sampling
    sample_chunks = random.sample(all_chunks, min(SAMPLE_SIZE, len(all_chunks)))
    print(f"🚀 Starting LLM Augmentation on {len(sample_chunks)} chunks...")

    augmented_data = []
    error_count = 0
    
    # 5. Main Loop
    for item in tqdm(sample_chunks, desc="Augmenting"):
        content = item.get('page_content', '')
        metadata = item.get('metadata', {})
        
        # Add Original Record
        augmented_data.append(item)
        
        # Generate Variations
        variations = await augment_chunk(content, llm)
        
        # RATE LIMITING: Add a small delay between requests to stay within free tier limits (15 RPM)
        # 15 RPM means 1 request every 4 seconds. Let's use 2.5s as a balanced delay.
        await asyncio.sleep(2.5)
        
        if not variations:
            error_count += 1
            
        for var in variations:
            new_item = {
                'page_content': var,
                'metadata': {**metadata, 'augmentation': 'llm_paraphrase'}
            }
            augmented_data.append(new_item)

    # 6. Save output
    output_file = 'synthetic_data_llm.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ LLM Augmentation complete.")
    print(f"   Results saved to {output_file}")
    
    if error_count > 0:
        print(f"⚠️ WARNING: {error_count} chunks failed to augment (check errors above).")

    # 7. Run Diagnostics
    analyze_start_phrases(output_file)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user.")