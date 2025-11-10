import os
import json
import re
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🧭", layout="centered")
st.title("SHL Assessment Recommender")
st.caption(f"Backend: {API_URL}")

debug = st.toggle("Show debug responses", value=False, help="Show raw API response and status codes")

def normalize_payload(data: dict):
    """
    Normalize backend response to:
    { "recommended_assessments": [...] }
    Accepts either {"recommended_assessments": [...]} or {"results": [...]}.
    """
    if not isinstance(data, dict):
        return None
    if "recommended_assessments" in data and isinstance(data["recommended_assessments"], list):
        return {"recommended_assessments": data["recommended_assessments"]}
    if "results" in data and isinstance(data["results"], list):
        return {"recommended_assessments": data["results"]}
    return None

def call_api(payload: dict):
    try:
        r = requests.post(f"{API_URL}/recommend", json=payload, timeout=120)

        if debug:
            st.write("Status:", r.status_code)
            try:
                st.write("Raw JSON:", r.json())
            except Exception:
                st.write("Raw text:", r.text)

        try:
            data = r.json()
        except Exception:
            return None, f"Non-JSON response from API (status={r.status_code}): {r.text[:500]}"

        if not r.ok:
            if isinstance(data, dict) and "detail" in data:
                return None, f"API error (status={r.status_code}): {data['detail']}"
            return None, f"HTTP {r.status_code}: {data}"

        normalized = normalize_payload(data)
        if normalized is None:
            return None, f"Unexpected response shape: {data}"

        return normalized, None

    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"

# ---------------- Single input that accepts text or URL (incl. hyperlinks) ----------------

def extract_url(text: str) -> str | None:
    """Return the first URL if the input contains one (plain, Markdown, or HTML)."""
    if not text:
        return None
    # Markdown: [label](https://...)
    m = re.search(r"\[[^\]]*\]\((https?://[^)\s]+)\)", text, re.I)
    if m:
        return m.group(1).strip()
    # HTML: <a href="https://...">
    m = re.search(r'href=["\'](https?://[^"\']+)["\']', text, re.I)
    if m:
        return m.group(1).strip()
    # Plain URL
    m = re.search(r"(https?://[^\s)]+)", text, re.I)
    if m:
        return m.group(1).strip()
    return None

user_input = st.text_area(
    "Enter a hiring query or paste a JD URL (Markdown/HTML links are OK)",
    height=180,
    placeholder="Examples:\n"
                "- Hiring a COO in China; English; ~60 mins; simulations preferred\n"
                "- https://example.com/jd.pdf\n"
                "- Find the JD [here](https://example.com/jd.html)",
)

if st.button("Recommend"):
    txt = (user_input or "").strip()
    if not txt:
        st.warning("Please enter a query or a JD URL.")
    else:
        url = extract_url(txt)
        payload = {"jd_url": url} if url else {"query": txt}
        with st.spinner("Recommending..."):
            result_json, err = call_api(payload)
        if err:
            st.error(err)
        else:
            st.success("Done")
            # Show clean JSON without Streamlit's list indices (0:, 1:)
            st.code(json.dumps(result_json, indent=2), language="json")
