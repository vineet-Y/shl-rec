# app/jd_fetch.py
import re, io, requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text

HEADERS = {"User-Agent": "Mozilla/5.0"}

def _is_pdf_url(url: str) -> bool:
    return bool(re.search(r"\.pdf($|[?#])", url, re.I))

def fetch_text_from_url(url: str, timeout: int = 20) -> str:
    url = url.strip()
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    content_type = r.headers.get("Content-Type","").lower()

    if _is_pdf_url(url) or "application/pdf" in content_type:
        # PDF path
        text = pdf_extract_text(io.BytesIO(r.content))
        return _clean_text(text)
    else:
        # HTML path
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        # crude readability: remove nav/script/style
        for tag in soup(["script","style","nav","footer","header","noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return _clean_text(text)

def _clean_text(t: str) -> str:
    t = t or ""
    t = re.sub(r"\s+", " ", t).strip()
    return t[:100000]  # hard cap to keep summaries stable
