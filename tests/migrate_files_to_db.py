"""Build comprehensive original-file-to-chunk mapping and store in PostgreSQL.
Simple approach: one UPDATE at a time with file caching."""
import sys, os, json, re
from urllib.parse import urlparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG Core"))
from rag.config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
import psycopg2, psycopg2.extras

def normalize_url(url):
    if not url:
        return ""
    url = url.strip().lower()
    if url.startswith("http://"):
        url = "https://" + url[7:]
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"https://{parsed.netloc}{path}"

BASE = os.path.join(os.path.dirname(__file__), "..", "RAG Core")

url_to_file = {}
pdf_map = {}
docx_map = {}

print("Scanning scraped_data_categorized...", flush=True)
for site in ["iit", "isb"]:
    cat_dir = os.path.join(BASE, site, "scraped_data_categorized")
    if not os.path.isdir(cat_dir):
        continue
    for root, dirs, files in os.walk(cat_dir):
        for f in files:
            full = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext == ".pdf":
                pdf_map[f.lower()] = full
            elif ext == ".docx":
                docx_map[f.lower()] = full
            elif ext == ".json":
                try:
                    with open(full, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    jurl = data.get("url", "")
                    if jurl:
                        norm = normalize_url(jurl)
                        url_to_file[norm] = (full, "application/json", f)
                except:
                    pass

print(f"  URL->JSON: {len(url_to_file)}, PDFs: {len(pdf_map)}, DOCXs: {len(docx_map)}", flush=True)

# Map ?p=NNNN URLs
p_pattern = re.compile(r"/\?p=(\d+)$")
for norm, (fp, ft, fn) in list(url_to_file.items()):
    m = p_pattern.search(norm)
    if m:
        base_url = norm.split("/?p=")[0]
        if base_url not in url_to_file:
            url_to_file[base_url] = (fp, ft, fn)

# Build upload:// lookup
upload_map = {}
for norm, (fp, ft, fn) in url_to_file.items():
    stem = os.path.splitext(fn)[0].lower()
    upload_map[stem] = (fp, ft, fn)

def get_file_path(url, site):
    """Map a chunk URL to (filepath, file_type, file_name) or None."""
    if not url:
        return None
    url_lower = url.lower()
    
    if url_lower.endswith(".pdf") and url_lower in pdf_map:
        fp = pdf_map[url_lower]
        return (fp, "application/pdf", os.path.basename(fp))
    elif url_lower.endswith(".docx") and url_lower in docx_map:
        fp = docx_map[url_lower]
        return (fp, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", os.path.basename(fp))
    elif url.startswith("http"):
        norm = normalize_url(url)
        if norm in url_to_file:
            return url_to_file[norm]
    elif url.startswith("C:") or url.startswith("D:"):
        if os.path.isfile(url):
            ext = os.path.splitext(url)[1].lower()
            ft = "application/pdf" if ext == ".pdf" else ("application/vnd.openxmlformats-officedocument.wordprocessingml.document" if ext == ".docx" else "text/plain; charset=utf-8")
            return (url, ft, os.path.basename(url))
    elif url.startswith("upload://"):
        inner = url[len("upload://"):]
        if inner.endswith(".txt"):
            inner = inner[:-4]
        stem = os.path.splitext(inner)[0].lower()
        if stem in upload_map:
            return upload_map[stem]
    return None

# Load all chunks
conn = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

print("Loading chunks...", flush=True)
cur.execute("SELECT id, url, page_content, title FROM chunks ORDER BY id")
rows = cur.fetchall()
print(f"Total chunks: {len(rows)}", flush=True)

# File read cache (key = filepath, value = binary bytes)
file_cache = {}
updated_original = 0
updated_txt = 0
not_found = 0
errors = 0

for r in rows:
    cid = r["id"]
    url = (r["url"] or "").strip()
    
    result = get_file_path(url, None)
    
    if result is not None:
        fp, ft, fn = result
        if fp not in file_cache:
            try:
                with open(fp, "rb") as fh:
                    file_cache[fp] = fh.read()
            except Exception as e:
                print(f"  Error reading {fp}: {e}", flush=True)
                file_cache[fp] = None
        binary = file_cache.get(fp)
        if binary is not None:
            try:
                cur.execute("UPDATE chunks SET file_data = %s::bytea, file_name = %s, file_type = %s WHERE id = %s",
                           (psycopg2.Binary(binary), fn, ft, cid))
                updated_original += 1
            except Exception as e:
                print(f"  DB error chunk {cid}: {e}", flush=True)
                errors += 1
        else:
            not_found += 1
    else:
        # Fallback: generate .txt from page_content
        content = (r["page_content"] or "").strip()
        if not content:
            content = f"Document: {r.get('title', 'Untitled')}\n\nNo content available."
        txt_bytes = content.encode("utf-8")
        title = (r["title"] or "document").strip()
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80].strip()
        fn = f"{safe_name}.txt"
        try:
            cur.execute("UPDATE chunks SET file_data = %s::bytea, file_name = %s, file_type = %s WHERE id = %s",
                       (psycopg2.Binary(txt_bytes), fn, "text/plain; charset=utf-8", cid))
            updated_txt += 1
        except Exception as e:
            print(f"  DB error txt chunk {cid}: {e}", flush=True)
            errors += 1
    
    if (updated_original + updated_txt) % 500 == 0:
        conn.commit()
        print(f"  progress: {updated_original} original + {updated_txt} txt = {updated_original + updated_txt} updated", flush=True)

conn.commit()
cur.close()
conn.close()

print(f"\nDone!", flush=True)
print(f"  Updated with original files: {updated_original}", flush=True)
print(f"  Updated with generated .txt:  {updated_txt}", flush=True)
print(f"  Not found/errors:            {not_found + errors}", flush=True)
print(f"  Total:                       {updated_original + updated_txt + not_found + errors}", flush=True)

# Final verification
conn2 = psycopg2.connect(host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD)
cur2 = conn2.cursor()
cur2.execute("SELECT COUNT(*) FROM chunks WHERE file_data IS NOT NULL")
has_file = cur2.fetchone()[0]
cur2.execute("SELECT COUNT(*) FROM chunks")
total = cur2.fetchone()[0]
cur2.execute("SELECT file_type, COUNT(*) FROM chunks WHERE file_data IS NOT NULL GROUP BY file_type ORDER BY COUNT(*) DESC")
types = cur2.fetchall()
cur2.close()
conn2.close()

print(f"\nVerification: {has_file}/{total} chunks have file_data", flush=True)
print("File types:", types, flush=True)
