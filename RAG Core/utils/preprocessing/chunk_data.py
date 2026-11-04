"""
chunk_data_v4.py — SemanticDataChunker (ratio bug fixed)
==========================================================
Root cause of the bug:
    V3 converted tokens → chars using a fixed ratio of 4.0 chars/token.
    French text has a real ratio of ~3.1 chars/token, so 756 chars of
    French = ~244 tokens, silently exceeding the 200-token target.

The fix:
    Use RecursiveCharacterTextSplitter.from_tiktoken_encoder() which
    measures chunk size directly in tokens via tiktoken internally.
    No ratio conversion needed. No approximation. Exact token counts.

Dependencies:
    pip install langchain-text-splitters tiktoken
"""

import re
import json
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Token counter (for metadata + breadcrumb measurement only) ─────────
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

    TIKTOKEN_AVAILABLE = True
    print("[chunker] tiktoken found — using exact token counting.")

except ImportError:
    def count_tokens(text: str) -> int:
        return len(text) // 4  # rough fallback

    TIKTOKEN_AVAILABLE = False
    print("[chunker] tiktoken not found — install with: pip install tiktoken")


class SemanticDataChunker:

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 30):
        """
        Args:
            chunk_size    : max tokens per chunk (default: 200)
            chunk_overlap : overlap tokens between chunks (default: 30)

        Both are now passed directly to LangChain's token-native splitter.
        No char conversion. No ratio approximation.
        """
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self.section_keywords = [
            "Programmes?", "Admissions?", "Objectifs?", "Points forts?",
            "Frais", "Contact", "Perspective", "A propos", "À propos",
            "Plan d'étude", "Plans d'études", "R&D", "Recherche",
            "Vie Estudiantine", "Stages?", "Projets?", "Les futurs métiers"
        ]

    # ── extract_metadata (unchanged) ───────────────────────────────────

    def extract_metadata(self, content: str, file_path) -> tuple[str, dict]:
        file_path = Path(file_path)

        category = "Unknown"
        for i, part in enumerate(file_path.parts):
            if "scraped_data_cleaned" in part and i + 1 < len(file_path.parts):
                category = file_path.parts[i + 1]
                break

        metadata = {
            "url"     : "Unknown",
            "title"   : file_path.stem.replace(".pdf", "").replace(".docx", "").replace("_", " "),
            "category": category,
            "site"    : "IIT" if "iit" in str(file_path).lower() else "ISB",
        }

        header_map = {
            "url"     : r"(?:Source URL|Source File):\s*(.*)",
            "title"   : r"Title:\s*(.*)",
            "category": r"Category:\s*(.*)",
            "site"    : r"Site:\s*(.*)",
        }
        for key, pattern in header_map.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()

        header_keys    = ["Source URL", "Source File", "Title", "Category", "Site", "Type"]
        header_pattern = r"^(?:" + "|".join(header_keys) + r"):\s*.*?\n"
        clean_content  = re.sub(
            header_pattern, "", content,
            flags=re.MULTILINE | re.IGNORECASE
        ).strip()

        return clean_content, metadata

    # ── get_sections (unchanged) ────────────────────────────────────────

    def get_sections(self, text: str) -> list[tuple[str, str]]:
        marker_pattern     = "|".join(self.section_keywords)
        section_regex_core = (
            fr"(?:^|\n)(?:(?:{marker_pattern})"
            fr"|[A-ZÉÀÈ\s]{{4,40}}"
            fr"|[A-ZÉÀÈ][^.\n]{{3,50}}:)"
        )
        full_pattern = f"(?im)({section_regex_core})"
        parts        = re.split(full_pattern, text)
        sections     = []

        if len(parts) == 1:
            content = parts[0].strip()
            if content:
                sections.append(("Introduction", content))
        else:
            if parts[0].strip():
                sections.append(("Introduction", parts[0].strip()))

            for i in range(1, len(parts), 2):
                header  = parts[i].strip().rstrip(":").strip()
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""

                if not content:
                    continue
                if len(re.sub(r'\W', '', content)) < 10:
                    continue

                sections.append((header, content))

        return sections

    # ── _make_splitter — THE FIX ────────────────────────────────────────

    def _make_splitter(self, breadcrumb: str) -> RecursiveCharacterTextSplitter:
        """
        Build a splitter whose budget accounts for the breadcrumb tokens.
        """
        breadcrumb_tokens = count_tokens(breadcrumb)

        # Guard: always leave at least 50 tokens for actual content
        available_tokens = max(50, self.chunk_size - breadcrumb_tokens)

        if TIKTOKEN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name = "cl100k_base",
                chunk_size    = available_tokens,
                chunk_overlap = self.chunk_overlap,
                separators    = ["\n\n", "\n", ". ", " ", ""],
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size    = available_tokens * 3,
                chunk_overlap = self.chunk_overlap * 3,
                separators    = ["\n\n", "\n", ". ", " ", ""],
            )

        return splitter

    # ── process_dir ────────────────────────────────────────────────────

    def process_dir(self, input_path: str) -> list[dict]:
        print(f"\nScanning {input_path}...")
        all_chunks: list[dict] = []
        files = list(Path(input_path).glob("**/*.txt"))
        print(f"Found {len(files)} file(s).")

        for f in files:
            try:
                raw       = f.read_text(encoding="utf-8", errors="ignore")
                clean_txt, meta = self.extract_metadata(raw, f)
                sections  = self.get_sections(clean_txt)

                base_breadcrumb = f"[{meta['site']} | {meta['title']}]"

                for head, content in sections:
                    current_breadcrumb = f"{base_breadcrumb} {head} > "
                    splitter   = self._make_splitter(current_breadcrumb)
                    sub_chunks = splitter.split_text(content)

                    for sc in sub_chunks:
                        if len(re.sub(r'\W', '', sc)) < 25:
                            continue

                        rich_chunk  = current_breadcrumb + sc
                        token_count = count_tokens(rich_chunk)

                        all_chunks.append({
                            "page_content": rich_chunk,
                            "metadata": {
                                "site"        : meta["site"],
                                "title"       : meta["title"],
                                "category"    : meta["category"],
                                "url"         : meta["url"],
                                "header"      : head,
                                "chunk_length": len(rich_chunk),
                                "token_count" : token_count,
                            },
                        })

            except Exception as e:
                print(f"  [error] {f.name}: {e}")

        print(f"  → {len(all_chunks)} chunks produced from {input_path}")
        return all_chunks


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunker = SemanticDataChunker(chunk_size=200, chunk_overlap=30)

    # Note: These paths assume you are running from the project root
    iit = chunker.process_dir("iit/scraped_data_cleaned")
    isb = chunker.process_dir("isb/scraped_data_cleaned")

    # Sanity check — token distribution
    all_chunks = iit + isb
    if all_chunks:
        counts = [c["metadata"]["token_count"] for c in all_chunks]
        over   = [c for c in counts if c > 200]
        print(f"\nToken count stats:")
        print(f"  min={min(counts)}  max={max(counts)}  avg={sum(counts)//len(counts)}")
        print(f"  chunks over 200 tokens: {len(over)} / {len(all_chunks)}")

    # Ensure data/raw directory exists
    os.makedirs("data/raw", exist_ok=True)

    with open("data/raw/iit_chunks.json", "w", encoding="utf-8") as f:
        json.dump(iit, f, ensure_ascii=False, indent=2)
    with open("data/raw/isb_chunks.json", "w", encoding="utf-8") as f:
        json.dump(isb, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Created {len(iit)} IIT and {len(isb)} ISB chunks in data/raw/")