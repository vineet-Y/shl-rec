import re
import time
import json
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

BASE = "https://www.shl.com"
CATALOG = "https://www.shl.com/products/product-catalog/"
STEP = 12  # SHL pagination increments by 12

# --- Test type mapping ---
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

# -------------------------------------------------
# PDF text extraction (visual order)
# -------------------------------------------------
def extract_text_in_visual_order_from_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out_lines = []
    for page in doc:
        blocks = page.get_text("blocks") or []
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))  # y, then x
        for b in blocks:
            txt = b[4]
            if txt:
                out_lines.append(txt.rstrip())
        out_lines.append("\n--- PAGE BREAK ---\n")
    raw = "\n".join(out_lines)
    raw = raw.replace("\u2022", "•")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    return raw.strip()


# Page scraping helpers

def _clean_lines(lines):
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()

def extract_page_text(psoup: BeautifulSoup) -> str:
    """Readable page text (fallback or for inspection)."""
    candidates = psoup.select(".product-catalogue, .content__container, main, article")
    container = candidates[0] if candidates else psoup.body or psoup
    for bad in container.select("script, style, nav, footer, form"):
        bad.decompose()
    lines = []
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li"], recursive=True):
        txt = el.get_text(" ", strip=True)
        if not txt: 
            continue
        if el.name == "li":
            lines.append("• " + txt)
        else:
            lines.append(txt)
    return _clean_lines(lines)

def extract_webpage_fields(psoup: BeautifulSoup):
    """
    Extract fields from an SHL assessment page.

    Returns:
      {
        "description": str|None,
        "job_levels": [str],
        "languages": [str],
        "test_type": [str],
        "duration": int|None,             # minutes
        "adaptive_support": "Yes"/"No",
        "remote_support": "Yes"/"No",
      }
    """
    fields = {
        "description": None,
        "job_levels": [],
        "languages": [],
        "test_type": [],
        "duration": None,
        "adaptive_support": "No",
        "remote_support": "No",
    }

    container = (
        psoup.select_one(".product-catalogue")
        or psoup.select_one(".content__container")
        or psoup
    )

    # ---------- Description ----------
    def _first_good_paragraph():
        for h in container.find_all(["h2", "h3", "h4"]):
            if re.search(r"\b(overview|about|description)\b", h.get_text(" ", strip=True), re.I):
                for sib in h.find_all_next(["p"], limit=6):
                    txt = sib.get_text(" ", strip=True)
                    if txt and len(txt) > 40:
                        return txt
        for p in container.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if txt and len(txt) > 40:
                return txt
        return None

    fields["description"] = _first_good_paragraph()

    # ---------- Job Levels (raw) ----------
    def _collect_job_levels():
        for h in container.find_all(["h3", "h4", "h5", "strong", "p"]):
            t = h.get_text(" ", strip=True)
            if re.match(r"(?i)^Job Level(s)?\b", t):
                ul = h.find_next_sibling(["ul", "ol"])
                if ul:
                    return [li.get_text(" ", strip=True) for li in ul.find_all("li")]
                m = re.search(r"Job Levels?:\s*([^\n]+)", t, re.I)
                if m:
                    return [x.strip() for x in re.split(r",|/|;", m.group(1)) if x.strip()]
                for sib in h.find_all_next(["p", "li"], limit=6):
                    txt = sib.get_text(" ", strip=True)
                    if txt and not re.match(r"(?i)^Job Level", txt):
                        return [x.strip() for x in re.split(r",|/|;", txt) if x.strip()]
        m = re.search(r"Job Levels?:\s*([^\n]+)", container.get_text("\n", strip=True), re.I)
        if m:
            return [x.strip() for x in re.split(r",|/|;", m.group(1)) if x.strip()]
        return []

    # Deduplicate but preserve original phrasing/case
    raw_job_levels = _collect_job_levels()
    seen = set()
    job_levels = []
    for lvl in raw_job_levels:
        if lvl not in seen:
            seen.add(lvl)
            job_levels.append(lvl)
    fields["job_levels"] = job_levels

    # ---------- Languages (inline only; not from Downloads) ----------
    def _inline_languages():
        langs = []

        # Label-led sections
        for h in container.find_all(["h3", "h4", "h5", "p", "strong"]):
            t = h.get_text(" ", strip=True)
            if re.match(r"(?i)^Languages?\b|^Available Languages?\b|^Language\b", t):
                ul = h.find_next_sibling(["ul", "ol"])
                if ul:
                    for li in ul.find_all("li"):
                        val = li.get_text(" ", strip=True)
                        if val and val not in langs:
                            langs.append(val)

                m = re.search(r"(?i)Languages?\s*:\s*([^\n]+)", t)
                if m:
                    for token in re.split(r",|/|;", m.group(1)):
                        token = token.strip()
                        if token and token not in langs:
                            langs.append(token)

                for sib in h.find_all_next(["p"], limit=3):
                    txt = sib.get_text(" ", strip=True)
                    if txt and not re.match(r"(?i)^Languages?\b|^Available Languages?\b|^Language\b", txt):
                        parts = [x.strip() for x in re.split(r",|/|;", txt) if x.strip()]
                        for p in parts:
                            if p not in langs:
                                langs.append(p)
                        break
                if langs:
                    return langs

        # Page-wide fallback like "Languages: English, French"
        m = re.search(r"(?i)Languages?\s*:\s*([^\n]+)", container.get_text("\n", strip=True))
        if m:
            for token in re.split(r",|/|;", m.group(1)):
                token = token.strip()
                if token and token not in langs:
                    langs.append(token)

        return langs

    fields["languages"] = _inline_languages()

    # ---------- Test Type (mapped) ----------
    tt_vals = []
    for h in container.find_all(["h3", "h4", "h5", "p", "strong"]):
        t = h.get_text(" ", strip=True)
        if re.match(r"(?i)^Test Type", t):
            m = re.search(r"Test Type[:\s]*([A-Z, /;+]+)", t, re.I)
            if m:
                tt_vals = [x.strip() for x in re.split(r"[,\s/;+]+", m.group(1)) if x.strip()]
                break
            sib = h.find_next(string=True)
            if sib:
                tt_vals = [x.strip() for x in re.split(r"[,\s/;+]+", str(sib)) if x.strip()]
                break

    mapped = []
    seen_tt = set()
    for code in tt_vals:
        up = code.upper()
        if len(up) == 1 and up in TEST_TYPE_MAP:
            full = TEST_TYPE_MAP[up]
        elif up in TEST_TYPE_MAP:
            full = TEST_TYPE_MAP[up]
        else:
            continue
        if full not in seen_tt:
            seen_tt.add(full)
            mapped.append(full)
    fields["test_type"] = mapped

    # ---------- Duration / Average completion time (minutes) ----------
    full_text = container.get_text("\n", strip=True)

    # strong match: "Assessment Length ... Approximate Completion Time in minutes = 22"
    m = re.search(
        r"Assessment\s*Length.*?(?:Approx(?:imate)?\s+)?(?:Completion|Testing)\s+Time\s*(?:in\s*minutes)?\s*(?:=|:)\s*(\d{1,3})",
        full_text,
        re.I | re.S
    )

    # general match
    if not m:
        m = re.search(
            r"(?:Approx(?:imate)?|Average)\s+(?:Completion|Testing)\s+Time(?:\s*\(minutes\))?\s*(?:=|:)?\s*(\d{1,3})",
            full_text,
            re.I
        )

    # loose fallback: any "<number> minutes"
    if not m:
        m = re.search(r"\b(\d{1,3})\s*(?:minutes?|mins?)\b", full_text, re.I)

    if m:
        try:
            fields["duration"] = int(m.group(1))
        except:
            pass

    # ---------- Adaptive support ----------
    
    adaptive_patterns = [
        r"\bcomputer[-\s]?adaptive\b",
        r"\badaptive testing\b",
        r"\badaptive assessment\b",
        r"\bCAT\b(?!\w)"  # CAT acronym
    ]
    if re.search("|".join(adaptive_patterns), full_text, re.I):
        fields["adaptive_support"] = "Yes"
    else:
        # simple 'adaptive' with test/assessment nearby
        if re.search(r"\badaptive\b.{0,40}\b(test|assessment|exam|questionnaire)\b", full_text, re.I):
            fields["adaptive_support"] = "Yes"

    # ---------- Remote support ----------
    # Heuristics: remote/virtual/online delivery/proctoring/administration
    remote_patterns = [
        r"\bremote(?:ly)?\b.{0,40}\b(proctor|proctoring|administration|delivery|testing|assessment)\b",
        r"\bvirtual\b.{0,40}\b(assessment|proctor|testing|delivery)\b",
        r"\bonline\b.{0,40}\b(assessment|proctor|testing|delivery)\b",
        r"\bunproctored online\b"
    ]
    if re.search("|".join(remote_patterns), full_text, re.I):
        fields["remote_support"] = "Yes"

    return fields


# HTTP helpers

def make_session():
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; SHL-FactSheet-Crawler/1.0)",
        "Accept-Language": "en-US,en;q=0.9"
    })
    return s

def fetch_html(session, url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def find_catalog_items(soup):
    links = []
    for a in soup.select('a[href*="/products/product-catalog/view/"]'):
        href = a.get("href")
        if href:
            links.append(urljoin(BASE, href))
    seen, out = set(), []
    for l in links:
        if l not in seen:
            seen.add(l); out.append(l)
    return out

def prefer_english(li):
    lang_tag = li.select_one(".product-catalogue__download-language")
    lang = lang_tag.get_text(strip=True).lower() if lang_tag else ""
    if "english (usa" in lang or "english (us" in lang: return 0
    if "english (uk" in lang or "english (gb" in lang: return 1
    if "english" in lang: return 2
    return 3

def find_fact_sheet_url(soup):
    candidates = []
    for li in soup.select("ul.product-catalogue__downloads li"):
        a = li.select_one("a[href]")
        if not a:
            continue
        text = a.get_text(strip=True).lower()
        if "product fact sheet" in text or "fact sheet" in text:
            url = urljoin(BASE, a["href"])
            candidates.append((prefer_english(li), url))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def get_assessment_name(soup):
    h1 = soup.select_one("h1")
    if h1: return h1.get_text(strip=True)
    if soup.title: return soup.title.get_text(strip=True)
    return None

def download_pdf(session, url):
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.content

# -------------------------------------------------
# Crawl logic

def crawl_to_json(out_json, out_txt_dir, start=1, max_pages=2, sleep=1.0):
    session = make_session()
    out_dir = Path(out_txt_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for page_num in range(max_pages):
        start_param = start + page_num * STEP
        page_url = f"{CATALOG}?start={start_param}&type=1&type=1"
        soup = fetch_html(session, page_url)
        detail_links = find_catalog_items(soup)
        if not detail_links:
            print(f"No links found at start={start_param}, stopping.")
            break

        print(f"🔍 Found {len(detail_links)} assessments on start={start_param}")

        for link in tqdm(detail_links, desc=f"Page start={start_param}"):
            try:
                time.sleep(sleep)
                psoup = fetch_html(session, link)
                name = get_assessment_name(psoup) or link

                # Extract from webpage
                fields = extract_webpage_fields(psoup)
                description   = fields.get("description")
                job_levels    = fields.get("job_levels") or []
                languages     = fields.get("languages") or []
                test_type_lst = fields.get("test_type") or []
                duration      = fields.get("duration")
                adaptive_yes  = fields.get("adaptive_support", "No")
                remote_yes    = fields.get("remote_support", "No")

                # Prefer PDF text from Product Fact Sheet if available
                pdf_url = find_fact_sheet_url(psoup)
                pdf_text = None
                if pdf_url:
                    try:
                        time.sleep(sleep)
                        pdf_bytes = download_pdf(session, pdf_url)
                        base_text = extract_text_in_visual_order_from_bytes(pdf_bytes) or ""
                        pdf_text = base_text.strip() if base_text else None

                        if pdf_text:
                            safe = re.sub(r"[^A-Za-z0-9_. -]+", "_", name)[:180]
                            (out_dir / f"{safe}.txt").write_text(pdf_text, encoding="utf-8")
                    except Exception:
                        pdf_text = None

                # If PDF not available, use "<name>. <description>" as pdf_text
                if not pdf_text:
                    desc_text = description.strip() if description else ""
                    name_part = name.strip()
                    combined_text = f"{name_part}. {desc_text}".strip()
                    pdf_text = combined_text if combined_text else None


                # Build record in requested schema
                record = {
                    "name": name,
                    "url": link,
                    "pdf_text": pdf_text,                                     # only from PDF
                    "job_level": ", ".join(job_levels) if job_levels else "", # string
                    "test_language": ", ".join(languages) if languages else "",# string
                    "adaptive_support": "Yes" if adaptive_yes == "Yes" else "No",
                    "description": description or "",
                    "duration": int(duration) if isinstance(duration, int) else (int(duration) if (isinstance(duration, str) and duration.isdigit()) else None),
                    "test_type": test_type_lst,                                # array of strings
                    "remote_support": "Yes" if remote_yes == "Yes" else "No",
                }
                results.append(record)

            except Exception as e:
                # Fallback blank record respecting schema
                results.append({
                    "name": name if 'name' in locals() else link,
                    "url": link,
                    "pdf_text": None,
                    "job_level": "",
                    "test_language": "",
                    "adaptive_support": "No",
                    "description": "",
                    "duration": None,
                    "test_type": [],
                    "remote_support": "No",
                })

    Path(out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Done. Wrote {len(results)} assessments to {out_json}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build SHL assessment dataset (JSON).")
    ap.add_argument("--out_json", default="shl_assessments.json")
    ap.add_argument("--out_txt_dir", default="fact_sheets_txt")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--max_pages", type=int, default=2,
                    help="Number of catalog pages (default 2 = start=1 and start=13)")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    crawl_to_json(
        out_json=args.out_json,
        out_txt_dir=args.out_txt_dir,
        start=args.start,
        max_pages=args.max_pages,
        sleep=args.sleep
    )


# I excuted python DataExtract.py --out_json Assessments_Dataset.json --out_txt_dir fact_sheets_txt --start=3 --max_pages=33 --sleep=1.0
# in terminal to extract the Dataset