"""Quick test: check source download_url"""
import json, urllib.request
body = json.dumps({"question": "ISB accreditation"}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8002/query", data=body, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=60)
result = json.loads(resp.read())
print("Sources:")
for s in result.get("sources", []):
    print(f'  {s["id"]}: url={s["url"]}')
    print(f'  site={s["site"]} title={s["title"]}')
print()
print("Answer (first 200):", result["answer"][:200])
