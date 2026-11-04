import os
import re
import json
import hashlib
import logging
from pathlib import Path
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# Force consistent results for language detection
DetectorFactory.seed = 0

# Professional Loaders from LangChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredHTMLLoader,
    UnstructuredODTLoader,
    UnstructuredRTFLoader,
    UnstructuredEmailLoader,
)

# Configuration relative to project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = ROOT_DIR / "scraped_data_categorized"
OUTPUT_DIR = ROOT_DIR / "scraped_data_cleaned"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Noise patterns: boilerplate / navigation / form elements / social media
# ---------------------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    # Navigation / UI chrome
    r"(?i)\bSkip\s+to\s+content\b",
    r"(?i)\bContinuer\s+la\s+lecture\b",
    r"(?i)\bGo\s+to(\s+the)?\s+(previous|next)\s+page\b",
    r"(?i)\bVous\s+devriez\s+également\s+aimer\b",
    r"(?i)\bPrenez\s+votre\s+avenir\s+en\s+main\b",
    r"(?i)\bAccès\s+rapide\b",
    r"(?i)\bEspace\s+étudiant\b",
    r"(?i)\bVidéothéque\b",
    r"(?i)\bVisite\s+virtuelle\b",
    r"(?i)\bVoir\s+plus\b",
    r"(?i)\bEn\s+savoir\s+plus\b",
    r"(?i)\bRead\s+More\b",
    r"(?i)\bSuivez-nous\s+sur\b",
    r"(?i)\bContactez-nous\b",
    r"(?i)\bActualités\b",
    # Contact / Form placeholders
    r"(?i)\bNom\s+et\s+prénom\b",
    r"(?i)\bVotre\s+email\b",
    r"(?i)\bNuméro\s+de\s+téléphone\b",
    r"(?i)\bFirst\s+Name\s+John\s+Last\s+Name\s+Smith\b",
    r"(?i)\bjohnsmith@example\.com\b",
    r"(?i)\bYour\s+email\b",
    r"(?i)\bemail\s+address\b",
    r"(?i)\bphone\s+number\b",
    r"(?i)\bJohn\s+Smith\b",
    # Brochure / marketing noise
    r"(?i)\bBrochure\s+Licen[sc]e\s+in\s+Business\s+Administration\b",
    r"(?i)\bThe\s+First\s+and\s+only\s+LBA\s+Program\s+in\s+Sfax\b",
    r"(?i)\bBrochure\s+Master\s+Digital\s+Marketing\b",
    r"(?i)\bDigitalisez\s+la\s+Réalité\b",
    r"(?i)\bBrochure\s+Licen[sc]e\s+Nationale\s+En\s+Gestion\b",
    r"(?i)\bBrochure\b",
    # Social media / Facebook feed noise
    r"(?i)\bSomething'?s\s+going\s+on(\s+at\s+NAU)?\b",
    r"(?i)\bWatch\s+(le\s+)?video\b",
    r"(?i)\bcomment\s+your\s+guess\b",
    r"(?i)\bfunniest\s+answers\b",
    r"(?i)\bGuessTheMoment\b",
    r"(?i)#NAU\b",
    r"(?i)#ISBSfax\b",
    r"(?i)#IIT\b",
    r"(?i)#Soirée",
    r"(?i)shorturl\.at",
    r"(?i)pre-inscription",
    # Social feed visual labels
    r"(?i)^\s*(Photo|Video|Link)\s*$",
    # Legal / footer boilerplate
    r"(?i)\bCopyright\b.*\bAll\s+rights\s+reserved\b",
    r"(?i)\bAll\s+rights\s+reserved\b",
    r"(?i)\bwww\.iit\.tn\b",
    r"(?i)\bisb\.tn\b",
    r"(?i)\bSfax\s+-\s+Tunisie\b",
    r"(?i)\bPage\s+\d+\s+sur\s+\d+\b",
    # ISB-specific meaningless strings
    r"(?i)\bInternational\s+École\s+of\s+Entreprise\b",
    r"(?i)\bInstitut\s+International\s+de\s+Technologie\b",
    r"(?i)\bAdmission\s+Espace\s+étudiant\b",
    r"(?i)\bonly\s+LBA\b",
    r"(?i)\bfirst\s+and\s+only\b",
    r"(?i)\bD'abord\s+et\s+seulement\b",
    r"(?i)\bDirectrice\s+du\s+programme\b",
]

# Words that almost always indicate an English sentence/fragment
ENGLISH_TRIGGERS = [
    r"\bthe\b", r"\band\b", r"\bwith\b", r"\bfrom\b", r"\bfor\b",
    r"\bthis\b", r"\bthat\b", r"\bare\b", r"\bwas\b", r"\bwere\b",
    r"\bhave\b", r"\bhas\b", r"\bby\b", r"\bat\b", r"\bof\b",
    r"\bin\b", r"\bto\b", r"\bis\b", r"\bit\b", r"\bour\b",
    r"\btheir\b", r"\bwe\b", r"\bthey\b", r"\byou\b",
    r"\bProgram\b", r"\bBusiness\b", r"\bUniversity\b", r"\bAlumni\b",
    r"\bCompetition\b", r"\byour email\b", r"\bfirst\b", r"\bonly\b",
    r"\bschool\b", r"\bmanagement\b", r"\bbusiness\b", r"\bMeet our\b",
    r"\bGo to\b", r"\bbusiness administration\b", r"\bstudents\b",
    r"\blearning\b", r"\bteaching\b", r"\bcourses\b", r"\btraining\b",
    r"\bsince\b", r"\bthus\b", r"\bless than\b", r"\bregarding\b",
    r"\bsuccessful\b", r"\benabled\b", r"\brecognized\b", r"\bwatch\b",
    r"\bcomment\b", r"\bguess\b", r"\bsomething's\b", r"\bgoing on\b",
    r"\bVideo\b", r"\bPhoto\b",
]


class DataProcessorLangChain:
    def __init__(self, site_name="IIT"):
        self.seen_hashes = set()
        self.site_name = site_name
        self.translator = GoogleTranslator(source='auto', target='fr')
        self.boilerplate_patterns = BOILERPLATE_PATTERNS
        self._en_triggers = ENGLISH_TRIGGERS

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------
    def _is_english(self, text):
        """Return True if the text is (likely) English."""
        if any(re.search(p, text, re.IGNORECASE) for p in self._en_triggers):
            return True
        if len(text) > 20:
            try:
                return detect(text) == 'en'
            except Exception:
                pass
        return False

    def _safe_translate(self, text):
        """Translate a single chunk; return original on any error."""
        try:
            result = self.translator.translate(text)
            return result if result else text
        except Exception:
            return text

    def translate_to_french(self, text):
        """
        Smart translation with two-pass strategy:
        1. If the WHOLE block is already French → return immediately (fast path).
        2. Otherwise, split on sentence delimiters and only translate
           segments that look English, batching adjacent English segments
           into a single API call to reduce network latency.
        """
        if not text:
            return ""

        # --- Fast path: entire block is French ---
        try:
            if len(text) > 30 and detect(text) == 'fr':
                return text
        except Exception:
            pass

        # --- Split on sentence-ending punctuation ---
        parts = re.split(r'([.!?…|])', text)
        translated_parts = []

        buffer_en = []   # pending English fragments to batch
        buffer_delim = []

        def flush_buffer():
            """Translate & flush accumulated English buffer."""
            if not buffer_en:
                return
            combined = " ".join(buffer_en)
            translated = self._safe_translate(combined)
            translated_parts.append(translated + "".join(buffer_delim))
            buffer_en.clear()
            buffer_delim.clear()

        for i in range(0, len(parts), 2):
            part = parts[i].strip()
            delim = parts[i + 1] if i + 1 < len(parts) else ""
            if not part:
                translated_parts.append(delim)
                continue

            if self._is_english(part):
                buffer_en.append(part)
                buffer_delim.append(delim)
            else:
                flush_buffer()
                translated_parts.append(part + delim)

        flush_buffer()
        return " ".join(translated_parts)

    # ------------------------------------------------------------------
    # Block-level deduplication (catches whole paragraph/section repeats)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(s):
        return re.sub(r'\W+', '', s.lower())

    @staticmethod
    def _deduplicate_blocks(text, min_words=20):
        """
        Greedy rolling-hash repeated-span removal.

        Works at the word level:
        1. Tokenise the cleaned text into words.
        2. Slide a window of `min_words` across the word list.
        3. For each window, compute a normalised hash of that span.
        4. If the hash was seen before, EXPAND the duplicate span as far as
           it matches, then mark that whole expanded range for removal.
        5. Rebuild the word list skipping all marked-for-removal positions.

        This reliably removes full paragraph blocks repeated 2–4× times
        (programme lists, "Des métiers…" blocks, etc.) regardless of whether
        they are surrounded by section headers or not.
        """
        # Tokenise
        words = text.split()
        n = len(words)
        if n < min_words * 2:
            return text

        # Normalise each word once for comparison
        norm_words = [re.sub(r'\W+', '', w.lower()) for w in words]

        # Build hash table: normalised span hash → first occurrence index
        def span_hash(start, length):
            return hashlib.md5(
                ' '.join(norm_words[start:start + length]).encode()
            ).hexdigest()

        first_seen = {}          # hash → first start index
        remove_mask = [False] * n  # True = word should be removed

        i = 0
        while i <= n - min_words:
            # Skip already-marked positions
            if remove_mask[i]:
                i += 1
                continue

            h = span_hash(i, min_words)
            if h in first_seen:
                first_start = first_seen[h]
                # The first occurrence starts at first_start; this is a duplicate.
                # Expand the match as far forward as possible.
                length = min_words
                while (i + length < n
                       and first_start + length < i   # don't overlap with original
                       and (norm_words[first_start + length]
                            == norm_words[i + length])):
                    length += 1

                # Mark duplicate for removal (i … i+length)
                for k in range(i, min(i + length, n)):
                    remove_mask[k] = True
                i += length
            else:
                first_seen[h] = i
                i += 1

        kept = [w for w, rm in zip(words, remove_mask) if not rm]
        return ' '.join(kept)

    # ------------------------------------------------------------------
    # Main cleaning pipeline
    # ------------------------------------------------------------------
    def clean_text(self, text):
        """Full noise-removal, deduplication, and translation pipeline."""
        if not text:
            return ""

        # 0. Basic normalization
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\s+', ' ', text).strip()

        # 1. Strip boilerplate patterns
        for pat in self.boilerplate_patterns:
            text = re.sub(pat, ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # 2. Translate English fragments → French
        text = self.translate_to_french(text)
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. Sentence-level filter + intra-sentence deduplicate
        blocklist = [
            r"Nom et prénom", r"Votre email", r"Numéro de téléphone",
            r"votre email", r"Your email", r"email address", r"phone number",
            r"John Smith", r"johnsmith", r"Continuer la lecture", r"Go to",
            r"Espace étudiant", r"Directrice du programme", r"Actualités",
            r"Vidéothéque", r"Visite virtuelle", r"Brochure",
            r"Program in", r"only LBA", r"first and only", r"LBA Sfax",
            r"D'abord et seulement", r"Skip to content",
            r"Vous devriez également aimer", r"Prenez votre avenir en main",
            r"Suivez-nous sur", r"Contactez-nous", r"Accès rapide",
            r"Copyright", r"All rights reserved",
            r"International École of Entreprise",
            r"Institut International de Technologie",
            r"Voir plus", r"En savoir plus", r"Read More",
            r"Something.s going on", r"Watch.*video", r"comment your guess",
            r"shorturl\.at", r"pre-inscription",
            r"Quickly aggregate B2B users", r"worldwide potentialities",
            r"Progressively plagiarize resource-leveling", r"resource-leveling core competencies",
            r"Dramatically mesh low-risk high-yield", r"transparent e-tailers",
        ]

        raw_chunks = re.split(r'(\. |\! |\? |… |\| )', text)
        clean_chunks = []
        seen_norms = []

        for i in range(0, len(raw_chunks), 2):
            chunk = raw_chunks[i].strip()
            delimiter = raw_chunks[i + 1] if i + 1 < len(raw_chunks) else ""
            if not chunk:
                continue

            # Blocklist check
            if any(re.search(p, chunk, re.IGNORECASE) for p in blocklist):
                continue

            # Collapse adjacent identical sub-phrases within chunk
            for _ in range(3):
                chunk = re.sub(
                    r'\b(.{10,}?)\s+\1\b', r'\1',
                    chunk, flags=re.IGNORECASE
                ).strip()

            norm = re.sub(r'\W+', '', chunk.lower())

            # Minimum useful length: 40 normalised chars
            if not norm or len(norm) < 40:
                continue

            # Sentence-level subsumption check
            is_redundant = False
            for prev_norm in seen_norms:
                if norm in prev_norm:
                    is_redundant = True
                    break

            if not is_redundant:
                to_remove = [
                    idx for idx, pn in enumerate(seen_norms) if pn in norm
                ]
                for idx in sorted(to_remove, reverse=True):
                    seen_norms.pop(idx)
                    clean_chunks.pop(idx)

                clean_chunks.append(chunk + delimiter)
                seen_norms.append(norm)

        text = " ".join(clean_chunks)
        text = re.sub(r'\s+', ' ', text).strip()

        # 4. Block-level paragraph deduplication (catches repeated sections)
        text = self._deduplicate_blocks(text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # ------------------------------------------------------------------
    # Redundancy gate (whole-document hash)
    # ------------------------------------------------------------------
    def is_redundant(self, text):
        """Return True if this document is near-empty or seen before."""
        semantic = re.sub(r'\W+', '', text.lower())
        if len(semantic) < 50:
            return True
        h = hashlib.md5(semantic.encode('utf-8')).hexdigest()
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    # ------------------------------------------------------------------
    # File processors
    # ------------------------------------------------------------------
    def _get_loader(self, in_path):
        """Return the appropriate LangChain loader for a given file path."""
        ext = os.path.splitext(in_path)[-1].lower()
        loaders = {
            '.pdf':   lambda p: PyPDFLoader(p),
            '.doc':   lambda p: Docx2txtLoader(p),
            '.docx':  lambda p: Docx2txtLoader(p),
            '.txt':   lambda p: TextLoader(p, encoding='utf-8'),
            '.csv':   lambda p: TextLoader(p, encoding='utf-8'),
            '.pptx':  lambda p: UnstructuredPowerPointLoader(p),
            '.ppt':   lambda p: UnstructuredPowerPointLoader(p),
            '.xlsx':  lambda p: UnstructuredExcelLoader(p),
            '.xls':   lambda p: UnstructuredExcelLoader(p),
            '.html':  lambda p: UnstructuredHTMLLoader(p),
            '.htm':   lambda p: UnstructuredHTMLLoader(p),
            '.odt':   lambda p: UnstructuredODTLoader(p),
            '.rtf':   lambda p: UnstructuredRTFLoader(p),
            '.eml':   lambda p: UnstructuredEmailLoader(p),
            '.msg':   lambda p: UnstructuredEmailLoader(p),
        }
        factory = loaders.get(ext)
        return factory(in_path) if factory else None

    def process_file(self, in_path, out_file_base):
        """Process any supported binary/text document."""
        filename = os.path.basename(in_path)
        loader = self._get_loader(in_path)
        if not loader:
            logging.warning(f"Unsupported file type, skipping: {filename}")
            return
        try:
            docs = loader.load()
            full_text = "\n".join(doc.page_content for doc in docs)
            cleaned = self.clean_text(full_text)

            if self.is_redundant(cleaned):
                logging.info(f"Skipping REDUNDANT / EMPTY file: {filename}")
                return

            header = (
                f"Source File: {filename}\n"
                f"Type: Document\n"
                f"Site: {self.site_name}\n\n"
            )
            out_path = out_file_base + ".txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(header + cleaned)
            logging.info(f"Cleaned (File): {os.path.basename(out_path)}")
        except Exception as e:
            logging.error(f"Error processing document {in_path}: {e}")

    def process_json_text(self, in_path, out_file_base):
        """Process a scraped HTML JSON file."""
        try:
            with open(in_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            raw_content = data.get("content", "")
            url = data.get("url", "")
            title = data.get("title", "No Title")

            cleaned = self.clean_text(raw_content)

            if self.is_redundant(cleaned):
                logging.info(f"Skipping REDUNDANT / EMPTY text: {title}")
                return

            header = (
                f"Source URL: {url}\n"
                f"Title: {title}\n"
                f"Category: Website Content\n"
                f"Site: {self.site_name}\n\n"
            )
            out_path = out_file_base + ".txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(header + cleaned)
            logging.info(f"Cleaned (HTML): {os.path.basename(out_path)}")
        except Exception as e:
            logging.error(f"JSON Error in {in_path}: {e}")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run(self, input_dir, output_dir):
        logging.info(
            f"--- Starting Advanced Data Cleaner (Target: {input_dir}) ---"
        )
        if not os.path.exists(input_dir):
            logging.error(f"No input data found in {input_dir}!")
            return

        os.makedirs(output_dir, exist_ok=True)

        for category in os.listdir(input_dir):
            cat_in = os.path.join(input_dir, category)
            if not os.path.isdir(cat_in):
                continue

            cat_out = os.path.join(output_dir, category)

            for sub in ["file", "text"]:
                folder_in = os.path.join(cat_in, sub)
                if not (os.path.exists(folder_in) and os.listdir(folder_in)):
                    continue
                folder_out = os.path.join(cat_out, sub)
                os.makedirs(folder_out, exist_ok=True)

                for fname in os.listdir(folder_in):
                    in_path = os.path.join(folder_in, fname)
                    if not os.path.isfile(in_path):
                        continue
                    out_base = os.path.join(folder_out, fname)
                    if sub == "file":
                        self.process_file(in_path, out_base)
                    else:
                        if fname.endswith('.json'):
                            self.process_json_text(in_path, out_base)

        logging.info(
            f"--- Cleaning/Translation Completed for {input_dir} ---"
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        site = sys.argv[3] if len(sys.argv) > 3 else "Unknown"
        cleaner = DataProcessorLangChain(site_name=site)
        cleaner.run(sys.argv[1], sys.argv[2])
    else:
        print(f"--- Running with defaults ---")
        print(f"Input: {INPUT_DIR}")
        print(f"Output: {OUTPUT_DIR}")
        site = "Unknown"
        cleaner = DataProcessorLangChain(site_name=site)
        cleaner.run(str(INPUT_DIR), str(OUTPUT_DIR))
