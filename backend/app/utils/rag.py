import os
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8002")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def _rewrite_file_url(url: str) -> str:
    """Rewrite relative /file URLs to absolute backend URLs."""
    if not url:
        return url
    if url.startswith("/"):
        return f"{BACKEND_URL}{url}"
    return url


async def ask_with_rag(question: str, top_k: int = 10, use_graph: bool = True) -> tuple[str, list]:
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(
            f"{RAG_API_URL}/query",
            json={"question": question, "top_k": top_k, "use_graph": use_graph},
        )
        resp.raise_for_status()
        data = resp.json()
        sources = data.get("sources", [])
        for s in sources:
            s["url"] = _rewrite_file_url(s.get("url", ""))
        return data.get("answer", ""), sources


async def generate_email(
    type_email: str, nom: str, prenom: str, universite: str, details: str = ""
) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{RAG_API_URL}/generate-email",
            json={
                "type_email": type_email,
                "nom": nom,
                "prenom": prenom,
                "universite": universite,
                "details": details,
            },
        )
        resp.raise_for_status()
        return resp.json()


def search_chunks(query: str, top_k: int = 20) -> list:
    import httpx
    resp = httpx.post(
        f"{RAG_API_URL}/query",
        json={"question": query, "top_k": top_k, "use_graph": True},
        timeout=180.0,
    )
    resp.raise_for_status()
    sources = resp.json().get("sources", [])
    for s in sources:
        s["url"] = _rewrite_file_url(s.get("url", ""))
    return sources
