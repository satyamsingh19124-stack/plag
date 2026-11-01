import os
import re
from datetime import datetime
from io import BytesIO

import docx
import nltk
import requests
import streamlit as st
import fitz  # PyMuPDF
from fpdf import FPDF
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ----------------------------- Page setup ------------------------------------
st.set_page_config(
    page_title="Verity | Plagiarism Check",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- Styling ---------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#182230; --muted:#687586; --line:#e7ebf1; --blue:#4263eb; --pale:#f5f7ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
.stApp { background: #f7f8fc; }
.block-container { max-width: 1180px; padding-top: 1.3rem; padding-bottom: 3rem; }
.brand { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.35rem; letter-spacing:-.6px; }
.brand span { color:#4263eb; }
.eyebrow { color:#4263eb; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; font-size:.72rem; }
.hero { background:linear-gradient(115deg,#172554 0%,#263d83 60%,#4263eb 100%);
        border-radius:22px; padding:34px 38px; color:white; margin: 12px 0 24px; }
.hero h1 { font-family:'Manrope',sans-serif; color:white; font-size:2.25rem; line-height:1.15; margin:.35rem 0 .65rem; }
.hero p { color:#dbe5ff; font-size:1rem; max-width:680px; margin:0; }
.panel { background:white; border:1px solid var(--line); border-radius:17px; padding:22px; }
.small-muted { color:var(--muted); font-size:.9rem; }
.metric-card { background:white; border:1px solid var(--line); border-radius:16px; padding:18px 20px; }
.metric-label { color:#687586; font-size:.83rem; font-weight:600; }
.metric-value { font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; color:#182230; margin-top:4px; }
.match-card { background:white; border:1px solid var(--line); border-radius:15px; padding:18px; margin:10px 0; }
.tag { display:inline-block; background:#eef2ff; color:#354fc5; padding:4px 9px; border-radius:20px; font-size:.75rem; font-weight:700; }
div.stButton > button { border-radius:10px; font-weight:700; min-height:2.8rem; }
div.stButton > button[kind="primary"] { background:#4263eb; border-color:#4263eb; }
div[data-testid="stFileUploader"] { background:#fbfcff; border:1px dashed #b9c5dc; border-radius:14px; padding:12px; }
hr { border-color:#e7ebf1; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- NLTK ------------------------------------------
@st.cache_resource
def prepare_nltk():
    for package, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
    return True

prepare_nltk()

# ----------------------------- Models ----------------------------------------
@st.cache_resource(show_spinner="Loading language model (first run may take a minute)...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------- Text extraction -------------------------------
def extract_text_from_pdf(file_obj):
    data = file_obj.read()
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

def extract_text_from_docx(file_obj):
    document = docx.Document(file_obj)
    return "\n".join(p.text for p in document.paragraphs)

def extract_text_from_txt(file_obj):
    raw = file_obj.read()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")

# ----------------------------- Search -----------------------------------------
def get_serpapi_results(query, api_key, num_results=5):
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results,
    }
    response = requests.get("https://serpapi.com/search", params=params, timeout=25)
    response.raise_for_status()
    data = response.json()
    unique_sources = {}
    for result in data.get("organic_results", []):
        snippet = result.get("snippet", "")
        url = result.get("link", "")
        if snippet and url and url not in unique_sources:
            unique_sources[url] = {
                "snippet": snippet,
                "url": url,
                "title": result.get("title", "Untitled source"),
            }
    return list(unique_sources.values())

def build_search_queries(text, max_queries=6):
    """Search several representative sentences instead of one document prefix."""
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.split()) >= 7]
    if not sentences:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return [cleaned[:220]] if cleaned else []
    # Prefer longer, information-rich sentences; keep query lengths manageable.
    ranked = sorted(sentences, key=lambda s: len(s.split()), reverse=True)
    queries, seen = [], set()
    for sentence in ranked:
        query = sentence[:240]
        key = re.sub(r"\W+", "", query.lower())
        if key and key not in seen:
            seen.add(key)
            queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries

def search_document(text, api_key):
    combined = {}
    for query in build_search_queries(text):
        for source in get_serpapi_results(query, api_key, num_results=5):
            combined[source["url"]] = source
    return list(combined.values())

# ----------------------------- Similarity ------------------------------------
def compute_cosine_similarity(input_text, sources):
    snippets = [src["snippet"] for src in sources]
    if not snippets:
        return []
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([input_text] + snippets)
        return cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError:
        return [0.0] * len(snippets)

def compute_semantic_similarity(input_text, sources, model):
    snippets = [src["snippet"] for src in sources]
    if not snippets:
        return []
    embeddings = model.encode([input_text] + snippets, convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1:]).cpu().numpy().flatten()

def calculate_plagiarism_score(cos_scores, sem_scores):
    if len(cos_scores) == 0 or len(sem_scores) == 0:
        return 0.0
    combined = 0.4 * cos_scores + 0.6 * sem_scores
    # This is a heuristic similarity indicator, not a verified plagiarism percentage.
    return float(min(100, max(0, combined.max() * 100)))

def get_top_matches(input_text, sources, model, top_n=10, threshold=0.48):
    input_sents = [s.strip() for s in sent_tokenize(input_text) if s.strip()]
    if not input_sents:
        return []
    all_matches = []
    input_embeds = model.encode(input_sents, convert_to_tensor=True)
    for src in sources:
        src_sents = [s.strip() for s in sent_tokenize(src["snippet"]) if s.strip()]
        if not src_sents:
            continue
        src_embeds = model.encode(src_sents, convert_to_tensor=True)
        sim_matrix = util.cos_sim(input_embeds, src_embeds)
        max_sim = sim_matrix.max(dim=1)
        for i, score_tensor in enumerate(max_sim.values):
            score = float(score_tensor.item())
            if score >= threshold:
                all_matches.append({
                    "input": input_sents[i],
                    "source_sent": src_sents[int(max_sim.indices[i].item())],
                    "score": score,
                    "source_url": src["url"],
                    "source_title": src["title"],
                })
    all_matches.sort(key=lambda item: item["score"], reverse=True)
    # Keep the strongest match per source URL, as in the supplied implementation.
    unique, seen = [], set()
    for match in all_matches:
        if match["source_url"] not in seen:
            seen.add(match["source_url"])
            unique.append(match)
        if len(unique) >= top_n:
            break
    return unique

# ----------------------------- PDF report -------------------------------------
def pdf_safe(value):
    return str(value).encode("latin-1", "replace").decode("latin-1")

def generate_pdf_report(plagiarism_score, matches):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "Plagiarism Check Report", ln=1, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 9, f"Estimated similarity indicator: {plagiarism_score:.1f}%", ln=1)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 6, "This is an automated estimate based on search snippets and text similarity. It is not a formal or exhaustive plagiarism determination.")
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Top source matches", ln=1)
    for idx, match in enumerate(matches, 1):
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 7, pdf_safe(f"Match #{idx} - semantic similarity: {match['score']:.2f}"))
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 6, pdf_safe(f"Source: {match['source_title']}"))
        pdf.multi_cell(0, 6, pdf_safe(f"URL: {match['source_url']}"))
        pdf.set_font("Arial", "B", 9)
        pdf.cell(0, 6, "Your content:", ln=1)
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 6, pdf_safe(match["input"]))
        pdf.set_font("Arial", "B", 9)
        pdf.cell(0, 6, "Matching source snippet:", ln=1)
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 6, pdf_safe(match["source_sent"]))
        pdf.ln(4)
    return bytes(pdf.output(dest="S"))

# ----------------------------- Header -----------------------------------------
left, right = st.columns([5, 1])
with left:
    st.markdown('<div class="brand">verity<span>.</span> <span style="color:#687586;font-size:.82rem;font-weight:500"> plagiarism intelligence</span></div>', unsafe_allow_html=True)
with right:
    st.markdown('<div style="text-align:right;color:#687586;font-size:.8rem;padding-top:8px">WEB + SEMANTIC SCAN</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow" style="color:#c7d4ff">Write with confidence</div>
  <h1>Know what matches.<br>Keep your work original.</h1>
  <p>Scan a document or paste your writing to discover similar passages across indexed web pages, review source matches, and export a clear report.</p>
</div>
""", unsafe_allow_html=True)

# API key: Streamlit secrets first, environment variable second.
try:
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
except Exception:
    SERPAPI_KEY = ""
SERPAPI_KEY = SERPAPI_KEY or os.getenv("SERPAPI_KEY", "")

upload_col, info_col = st.columns([1.35, 0.8], gap="large")
with upload_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### 1. Add your document")
    st.caption("Upload a PDF, Word document, or plain-text file.")
    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    pasted_text = st.text_area("Or paste your text here", height=190, placeholder="Paste the content you want to check...")
    st.markdown('</div>', unsafe_allow_html=True)

with info_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### What gets checked?")
    st.markdown("""
    - **Web matches** from Google-indexed pages
    - **Wording similarity** using TF-IDF
    - **Semantic similarity** using a sentence-transformer model
    - **Sentence-level comparisons** with source links
    """)
    st.info("Search coverage depends on the public pages indexed by Google and the snippets returned by the search API.")
    st.markdown('</div>', unsafe_allow_html=True)

input_text = ""
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            input_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.lower().endswith(".docx"):
            input_text = extract_text_from_docx(uploaded_file)
        else:
            input_text = extract_text_from_txt(uploaded_file)
        if pasted_text.strip():
            st.warning("Both a file and pasted text are present. The uploaded file will be checked.")
        st.success(f"Document loaded: {uploaded_file.name} · {len(input_text.split())} words")
    except Exception as exc:
        st.error(f"Could not read this document: {exc}")
elif pasted_text.strip():
    input_text = pasted_text

st.markdown("")
run_col, note_col = st.columns([1, 3])
with run_col:
    run_check = st.button("🔎  Check for plagiarism", type="primary", use_container_width=True)
with note_col:
    st.caption("Your document is processed for this scan. This demo does not save uploaded documents to a database.")

if run_check:
    if not input_text.strip():
        st.warning("Upload a document or paste some text before starting the scan.")
    elif not SERPAPI_KEY:
        st.error("SerpAPI key is missing. Add SERPAPI_KEY to .streamlit/secrets.toml or your environment variables.")
    else:
        with st.spinner("Searching web sources and comparing passages..."):
            try:
                sources = search_document(input_text, SERPAPI_KEY)
                if not sources:
                    st.warning("No web results were found for the selected passages. Try a longer or more distinctive document.")
                else:
                    model = load_model()
                    cos_scores = compute_cosine_similarity(input_text, sources)
                    sem_scores = compute_semantic_similarity(input_text, sources, model)
                    plagiarism_score = calculate_plagiarism_score(cos_scores, sem_scores)
                    top_matches = get_top_matches(input_text, sources, model)

                    st.session_state["scan_result"] = {
                        "score": plagiarism_score,
                        "matches": top_matches,
                        "source_count": len(sources),
                        "word_count": len(input_text.split()),
                    }
            except requests.exceptions.RequestException as exc:
                st.error(f"Search request failed: {exc}")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

# Persist results across Streamlit reruns (e.g. clicking download).
if "scan_result" in st.session_state:
    result = st.session_state["scan_result"]
    st.markdown("---")
    st.markdown('<div class="eyebrow">Scan complete</div>', unsafe_allow_html=True)
    st.markdown("## Your originality overview")
    score = result["score"]
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Similarity indicator</div><div class="metric-value">{score:.1f}%</div><div class="small-muted">Estimated from returned sources</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sources reviewed</div><div class="metric-value">{result["source_count"]}</div><div class="small-muted">Unique web pages returned</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Document length</div><div class="metric-value">{result["word_count"]:,}</div><div class="small-muted">Words analyzed</div></div>', unsafe_allow_html=True)

    if score >= 70:
        st.error("High similarity signal — review the matching passages and sources below.")
    elif score >= 40:
        st.warning("Some notable similarities were found — review the source matches.")
    else:
        st.success("A relatively low similarity signal was found in the sources checked.")

    st.markdown("### Matching passages")
    matches = result["matches"]
    if matches:
        for idx, match in enumerate(matches, 1):
            st.markdown('<div class="match-card">', unsafe_allow_html=True)
            st.markdown(f"**Match {idx}** &nbsp; <span class='tag'>{match['score']*100:.0f}% semantic similarity</span>", unsafe_allow_html=True)
            st.markdown(f"[{match['source_title']}]({match['source_url']})")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Your document**")
                st.warning(match["input"])
            with c2:
                st.markdown("**Web result snippet**")
                st.info(match["source_sent"])
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No sentence-level matches crossed the similarity threshold. This does not guarantee that the document is plagiarism-free.")

    st.markdown("### Export your report")
    report = generate_pdf_report(score, matches)
    st.download_button(
        "⬇️ Download PDF report",
        data=report,
        file_name=f"verity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        type="primary",
    )

st.markdown("---")
st.caption("Verity is an assistance tool, not an academic misconduct verdict. Search snippets are incomplete, and similarity can occur in quotations, references, common phrases, or properly attributed material.")
