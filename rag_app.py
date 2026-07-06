"""
PhenoPrompt — Clinical Phenotype Explorer
Same visual design as the MIMIC dashboard, powered entirely by the PhenoPrompt
pipeline outputs (AGBonnet synthetic notes → medkit NER → LSA/UMAP/HDBSCAN →
entity-augmented RAG). Reads data from data/phenoprompt/ in this repo.
"""
import json, math, re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR   = Path(__file__).parent / "data" / "phenoprompt"
STAGE1_DIR = DATA_DIR / "stage1_outputs"
STAGE2_DIR = DATA_DIR / "stage2_outputs"

CHAT_MODEL = "mistral-small-latest"

# Fixed Stage-2 scores from the full 28.5k normalized run (not stored in files)
SILHOUETTE = 0.669
DBCV       = 0.123

# ── Page config + styling (matches MIMIC dashboard) ───────────────────────────
st.set_page_config(page_title="PhenoPrompt", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .main { background: radial-gradient(circle at 15% 0%, #0d2030 0%, #0a1822 55%, #081019 100%); }
  h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; color: #eaf4f7; }
  .hero-title { font-family:'Fraunces',serif; font-size:2.6rem; font-weight:600;
                background:linear-gradient(92deg,#54C285,#1FA6C9 55%,#57C8B9);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:0.1rem; }
  .hero-sub { color:#8fb2c0; font-size:1.02rem; margin-top:0; font-family:'IBM Plex Mono',monospace; }
  .metric-card { background:rgba(31,166,201,0.07); border:1px solid rgba(84,194,133,0.22);
                 border-radius:14px; padding:1rem 1.2rem; }
  .metric-val { font-family:'Fraunces',serif; font-size:2rem; color:#54C285; font-weight:600; }
  .metric-lbl { color:#8fb2c0; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; }
  .chip { display:inline-block; padding:3px 11px; margin:2px; border-radius:999px;
          font-family:'IBM Plex Mono',monospace; font-size:0.78rem; border:1px solid; }
  .stTabs [data-baseweb="tab-list"] { gap:4px; }
  .stTabs [data-baseweb="tab"] { background:rgba(255,255,255,0.03); border-radius:10px 10px 0 0; padding:8px 18px; }
  [data-testid="stSidebar"] { background:#081019; border-right:1px solid rgba(84,194,133,0.15); }
</style>
""", unsafe_allow_html=True)

SYNONYMS = {
    "type 2 diabetes": ["diabetes"], "t2dm": ["diabetes"], "diabetic": ["diabetes"],
    "renal": ["kidney", "renal failure", "chronic kidney disease"],
    "kidney": ["renal failure", "chronic kidney disease"], "ckd": ["chronic kidney disease"],
    "hf": ["heart failure"], "chf": ["heart failure"], "sob": ["shortness of breath"],
    "breathless": ["shortness of breath"], "fluid overload": ["edema"], "swelling": ["edema"],
    "diuretic": ["furosemide"], "lung infection": ["pneumonia"], "chest infection": ["pneumonia"],
    "respiratory infection": ["pneumonia"], "infection": ["pneumonia"],
}
SYSTEM = ("You are a careful clinical informatics assistant answering questions about a cohort of "
          "patient notes. Use ONLY the numbered notes provided as context. Cite the notes you use "
          "with their id in square brackets like [note 1234]. If the notes do not contain enough "
          "information to answer, say so plainly. Do NOT invent diagnoses, drugs, values, or guidance.")
SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_clinical_notes",
        "description": ("Search the clinical-note corpus for notes relevant to a clinical concept "
                        "(condition, medication, symptom, or finding). Call this to ground your answer."),
        "parameters": {"type": "object", "properties": {"query": {"type": "string",
            "description": ("Concise clinical search phrase; correct spelling and extract the "
                            "conditions/medications from the user's wording.")}}, "required": ["query"]},
    }}]

def get_key():
    import os
    k = os.environ.get("MISTRAL_API_KEY")
    if k: return k
    try: return st.secrets["MISTRAL_API_KEY"]
    except Exception: return None

@st.cache_data(show_spinner="Loading PhenoPrompt data …")
def load_data():
    out = {}
    ncsv = STAGE1_DIR / "notes.csv.gz"
    out["notes"] = (dict(zip(*[pd.read_csv(ncsv, dtype={"idx": str}, compression="gzip")[c]
                    for c in ["idx", "note"]])) if ncsv.exists() else {})
    mcsv = STAGE1_DIR / "entity_mentions.csv"
    out["mentions"] = pd.read_csv(mcsv, dtype={"note_id": str}) if mcsv.exists() else pd.DataFrame()
    ccsv = STAGE2_DIR / "cluster_assignments.csv"
    out["clusters"] = pd.read_csv(ccsv, dtype={"note_id": str}) if ccsv.exists() else pd.DataFrame()
    ucsv = STAGE2_DIR / "umap_2d_coords.csv"
    out["umap"] = pd.read_csv(ucsv, dtype={"note_id": str}) if ucsv.exists() else pd.DataFrame()
    pj = STAGE2_DIR / "phenotype_profiles.json"
    out["profiles"] = json.loads(pj.read_text()) if pj.exists() else {}
    return out

@st.cache_data(show_spinner=False)
def build_retrieval_index():
    """Per-note entity lists + idf, for RAG retrieval."""
    D = load_data(); m = D["mentions"]; notes = D["notes"]
    if m.empty or not notes: return {}, {}, set()
    col = "text" if "text" in m.columns else "concept"
    if "assertion" in m.columns:
        m = m[m["assertion"] == "affirmed"]
    m = m[m["note_id"].isin(notes)]
    g = m.groupby("note_id")[col].apply(list)
    note_ents = {nid: g.get(nid, []) for nid in notes}
    dfc = {}
    for ents in note_ents.values():
        for e in set(ents): dfc[e] = dfc.get(e, 0) + 1
    N = max(len(notes), 1)
    idf = {e: math.log((N + 1) / (c + 1)) + 1 for e, c in dfc.items()}
    return note_ents, idf, set(idf)

def query_entities(q, vocab):
    q = q.lower(); hits = {}
    for e in vocab:
        if e in q: hits[e] = max(hits.get(e, 0.0), 1.0)
    for syn, tgts in SYNONYMS.items():
        if syn in q:
            for t in tgts:
                if t in vocab: hits[t] = max(hits.get(t, 0.0), 0.9)
    qtokens = [w for w in re.findall(r"[a-z]+", q) if len(w) >= 4]
    for e in vocab:
        if set(e.split()) & set(qtokens): hits[e] = max(hits.get(e, 0.0), 0.6)
    return hits

def retrieve(question, note_ents, idf, vocab, k):
    qe = query_entities(question, vocab)
    if not qe: return []
    scored = []
    for nid, ents in note_ents.items():
        if not ents: continue
        tf = {}
        for e in ents: tf[e] = tf.get(e, 0) + 1
        s = sum(w * tf.get(e, 0) * idf.get(e, 1.0) for e, w in qe.items())
        if s > 0: scored.append((nid, round(float(s), 3)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

def llm_answer(question, notes, hits, key):
    blocks = [f"[note {nid}]\n{notes.get(nid,'')[:1200]}" for nid, _ in hits]
    context = "\n\n".join(blocks)
    if not key:
        return "**No MISTRAL_API_KEY set** — showing retrieved notes only.\n\n" + "\n\n".join(
            f"[note {nid}] (score {s})\n\n{notes.get(nid,'')[:400]}" for nid, s in hits)
    from mistralai import Mistral
    cl = Mistral(api_key=key)
    r = cl.chat.complete(model=CHAT_MODEL, temperature=0.1, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context notes:\n\n{context}\n\nQuestion: {question}"}])
    return r.choices[0].message.content

def agentic_answer(question, notes, note_ents, idf, vocab, key, k):
    from mistralai import Mistral
    client = Mistral(api_key=key)
    messages = [
        {"role": "system", "content": SYSTEM + " You have a tool to search the clinical notes. "
         "Always call it before answering, normalising the user's wording into clean clinical terms."},
        {"role": "user", "content": question}]
    r1 = client.chat.complete(model=CHAT_MODEL, messages=messages, tools=SEARCH_TOOL,
                              tool_choice="any", temperature=0.1)
    msg = r1.choices[0].message
    if not getattr(msg, "tool_calls", None):
        hits = retrieve(question, note_ents, idf, vocab, k)
        return llm_answer(question, notes, hits, key), hits, question, False
    tc = msg.tool_calls[0]
    try: sq = json.loads(tc.function.arguments)["query"]
    except Exception: sq = question
    hits = retrieve(sq, note_ents, idf, vocab, k)
    tool_content = "\n\n".join(f"[note {nid}]\n{notes.get(nid,'')[:1200]}" for nid, _ in hits) or "No notes."
    messages.append(msg)
    messages.append({"role": "tool", "name": tc.function.name, "content": tool_content, "tool_call_id": tc.id})
    r2 = client.chat.complete(model=CHAT_MODEL, messages=messages, temperature=0.1)
    return r2.choices[0].message.content, hits, sq, True

# ── Load ──────────────────────────────────────────────────────────────────────
D = load_data()
clusters_df, umap_df, mentions, profiles, notes = (
    D["clusters"], D["umap"], D["mentions"], D["profiles"], D["notes"])
note_ents, idf, vocab = build_retrieval_index()
key = get_key()

n_notes = clusters_df["note_id"].nunique() if not clusters_df.empty else len(notes)
if not clusters_df.empty:
    lab = clusters_df["cluster"].astype(int)
    n_clusters = lab[lab != -1].nunique(); noise_frac = (lab == -1).mean()
    sizes = lab.value_counts().sort_index().to_dict()
else:
    n_clusters, noise_frac, sizes = 0, 0.0, {}
n_concepts = (mentions.loc[mentions.get("assertion","")=="affirmed","concept"].nunique()
              if "concept" in mentions.columns else len(vocab))

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Clinical Phenotype Explorer</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Discover patient subgroups from clinical notes — '
            'and query them in natural language</p>', unsafe_allow_html=True)
st.write("")
st.info("👉 Open the **🤖 PhenoPrompt** tab to ask a question in plain English.")

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{n_notes:,}", "Notes"),
    (c2, f"{n_concepts}", "Concepts"),
    (c3, f"{n_clusters}", "Phenotypes"),
    (c4, f"{SILHOUETTE:.3f}", "Silhouette")]:
    col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                 f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)
st.write("")

PALETTE = ["#54C285","#1FA6C9","#57C8B9","#9B5DE5","#F4CC47","#E2725B","#FF7F50","#8D99AE"]

tab_map, tab_clusters, tab_concepts, tab_dq, tab_data, tab_rag = st.tabs(
    ["🗺️ Phenotype Map", "🧬 Cluster Profiles", "📊 Top Concepts",
     "✅ Data Quality", "🗃️ Notes Table", "🤖 PhenoPrompt"])

# --- Phenotype Map (with entity-based overlay) ---
with tab_map:
    st.subheader("Phenotype map")
    if umap_df.empty or clusters_df.empty:
        st.info("umap or cluster files not found.")
    else:
        _u = umap_df.drop(columns=[c for c in ["cluster"] if c in umap_df.columns])
        m = _u.merge(clusters_df, on="note_id", how="left")
        m["cluster"] = m["cluster"].fillna(-1).astype(int)
        m["group"] = m["cluster"].apply(lambda c: "noise" if c == -1 else f"cluster {c}")
        # entity-based overlay: colour by number of entities per note (proxy for a clinical variable)
        overlay = st.selectbox("Colour by", ["Phenotype cluster", "Entity count per note"])
        if overlay == "Entity count per note" and note_ents:
            m["entity_count"] = m["note_id"].map(lambda i: len(note_ents.get(i, [])))
            fig = px.scatter(m, x="x", y="y", color="entity_count",
                             color_continuous_scale="Viridis", opacity=0.7, height=600)
        else:
            fig = px.scatter(m, x="x", y="y", color="group",
                             color_discrete_sequence=PALETTE, opacity=0.7, height=600)
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#eaf4f7", legend_title_text="")
        st.plotly_chart(fig, width='stretch')
        st.caption(f"{n_notes:,} notes · {n_clusters} clusters · {noise_frac*100:.0f}% noise")

# --- Cluster Profiles ---
with tab_clusters:
    st.subheader("Cluster profiles")
    if not profiles:
        st.info("phenotype_profiles.json not found.")
    else:
        keys = sorted([k for k in profiles if k.lstrip("-").isdigit()], key=lambda x: int(x))
        for k in keys:
            p = profiles[k]; top = p.get("top_entities", [])[:8]
            chips = "".join(f'<span class="chip" style="color:{PALETTE[i%len(PALETTE)]};'
                            f'border-color:{PALETTE[i%len(PALETTE)]}">{e.get("entity","?")}</span>'
                            for i, e in enumerate(top))
            st.markdown(f"**Cluster {p.get('cluster_id', k)}** · {p.get('n_notes','?')} notes",
                        unsafe_allow_html=True)
            st.markdown(chips, unsafe_allow_html=True); st.write("")

# --- Top Concepts ---
with tab_concepts:
    st.subheader("Most frequent clinical concepts")
    if mentions.empty:
        st.info("entity_mentions.csv not found.")
    else:
        col = "concept" if "concept" in mentions.columns else "text"
        aff = mentions[mentions.get("assertion","affirmed")=="affirmed"] if "assertion" in mentions.columns else mentions
        top = aff[col].value_counts().head(20).reset_index(); top.columns = ["concept", "count"]
        fig = px.bar(top, x="count", y="concept", orientation="h", height=560,
                     color_discrete_sequence=["#54C285"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#eaf4f7", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')

# --- Data Quality tab ---
with tab_dq:
    st.subheader("Clustering quality & data health")
    q = st.columns(4)
    total = sum(sizes.values()) if sizes else 0
    noise = sizes.get(-1, 0)
    q[0].metric("Notes clustered", f"{total-noise:,}")
    q[1].metric("Noise notes", f"{noise:,}")
    q[2].metric("Silhouette", f"{SILHOUETTE:.3f}")
    q[3].metric("DBCV", f"{DBCV:.3f}")
    st.caption(f"Silhouette ≈ {SILHOUETTE} and DBCV ≈ {DBCV} from the full 28,562-note normalized "
               f"run. Noise fraction ≈ {noise_frac*100:.0f}% reflects entity sparsity per note — "
               "the main limitation and focus of ongoing work.")
    if sizes:
        sd = (pd.DataFrame({"cluster": list(sizes.keys()), "notes": list(sizes.values())})
              .replace({"cluster": {-1: "noise"}}).set_index("cluster"))
        st.bar_chart(sd, height=300)

# --- Notes Table ---
with tab_data:
    st.subheader("Notes & their clusters")
    if clusters_df.empty:
        st.info("cluster_assignments.csv not found.")
    else:
        v = clusters_df.copy(); v["cluster"] = v["cluster"].astype(int)
        if notes:
            v["note_preview"] = v["note_id"].map(lambda i: (notes.get(i,"")[:160]+"…") if notes.get(i) else "")
        st.dataframe(v.head(500), width='stretch', height=520)
        st.caption(f"Showing first 500 of {len(v):,} notes.")

# --- PhenoPrompt RAG (Standard + Agentic) ---
with tab_rag:
    st.subheader("🤖 PhenoPrompt — ask a question")
    cc1, cc2 = st.columns([3, 1])
    with cc2:
        mode = st.radio("Mode", ["Standard RAG", "Agentic RAG"], index=0,
                        help="Agentic mode lets the model reformulate your question before searching.")
        topk = st.slider("Notes to retrieve", 3, 15, 5)
        st.caption(f"Mistral key: {'set ✓' if key else 'missing ✗'}")
    with cc1:
        q = st.text_input("Clinical question",
                          "What medications are documented for patients with diabetes and kidney disease?")
        ask = st.button("Ask", type="primary")
    if ask and q:
        if not vocab:
            st.info("Retrieval index empty (entity_mentions.csv not loaded).")
        else:
            agentic = mode.startswith("Agentic") and key
            if agentic:
                with st.spinner("Agent reformulating, searching, answering…"):
                    try:
                        resp, hits, sq, was_agentic = agentic_answer(q, notes, note_ents, idf, vocab, key, topk)
                        if was_agentic: st.info(f"🔧 The agent searched for: **{sq}**")
                    except Exception as e:
                        st.warning(f"Agentic failed ({e}); using standard.")
                        hits = retrieve(q, note_ents, idf, vocab, topk); resp = llm_answer(q, notes, hits, key)
            else:
                hits = retrieve(q, note_ents, idf, vocab, topk)
                with st.spinner("Retrieving and answering…"):
                    resp = llm_answer(q, notes, hits, key)
            if not hits:
                st.warning("No relevant notes retrieved. Try different terms.")
            else:
                st.markdown("### Answer"); st.markdown(resp)
                st.markdown("**Source notes:** " + ", ".join(str(n) for n, _ in hits))
                st.markdown("### Retrieved evidence")
                for nid, sc in hits:
                    with st.expander(f"note {nid} · score {sc}"):
                        st.write(notes.get(nid, ""))

# ── Sidebar (display controls, not live pipeline) ─────────────────────────────
with st.sidebar:
    st.markdown("### PhenoPrompt")
    st.caption("Prompt-based clinical phenotype discovery on synthetic notes.")
    st.markdown(f"- **{n_notes:,}** notes")
    st.markdown(f"- **{n_concepts}** concepts")
    st.markdown(f"- **{n_clusters}** phenotype clusters")
    st.markdown(f"- **{noise_frac*100:.0f}%** noise")
    st.markdown(f"- silhouette **{SILHOUETTE}**")
    st.markdown("---")
    st.caption("Note: clusters are pre-computed (Stage 2). This app displays "
               "results; it does not re-run clustering live.")
    st.caption("Data: AGBonnet synthetic clinical notes. Reproducible & open.")
