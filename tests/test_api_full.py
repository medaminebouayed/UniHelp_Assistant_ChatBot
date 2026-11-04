"""Test API: start server, query, download file"""
import subprocess, sys, os, json, urllib.request, time

api_dir = os.path.join(os.path.dirname(__file__), "..", "RAG Core")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "rag_api.main:app", "--port", "8002", "--host", "0.0.0.0"],
    cwd=api_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

for i in range(30):
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8002/health", timeout=3)
        if resp.status == 200:
            print("API started")
            break
    except:
        pass
    time.sleep(1)
else:
    print("API failed to start")
    proc.kill()
    sys.exit(1)

try:
    body = json.dumps({"question": "ISB accreditation"}).encode()
    req = urllib.request.Request("http://127.0.0.1:8002/query", data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    
    print("Sources:")
    for s in result["sources"]:
        print(f'  {s["id"]}: url={s["url"]}')
    print()
    
    # Download first file
    first_url = result["sources"][0]["url"]
    full_url = "http://127.0.0.1:8002" + first_url
    print(f"Downloading: {full_url}")
    dreq = urllib.request.Request(full_url)
    dresp = urllib.request.urlopen(dreq, timeout=30)
    data = dresp.read()
    print(f"Download OK - Type: {dresp.headers.get('Content-Type')}, Size: {len(data)} bytes")
    ddisp = dresp.headers.get('Content-Disposition')
    print(f"Content-Disposition: {ddisp[:120] if ddisp else 'N/A'}")
    print(f"First 100 bytes: {data[:100]}")
    
finally:
    proc.kill()
    proc.wait()
    print("API stopped")
