"""
PhenoPrompt — Clinical Phenotype Explorer
Same visual design as the MIMIC dashboard, but powered entirely by the
PhenoPrompt pipeline outputs (AGBonnet synthetic notes → medkit NER →
LSA/UMAP/HDBSCAN clustering → entity-augmented RAG).

Reads data from data/phenoprompt/ committed in this repo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data" / "phenoprompt"
STAGE1_DIR = DATA_DIR / "stage1_outputs"
STAGE2_DIR = DATA_DIR / "stage2_outputs"

# ── Page config + shared styling (matches the MIMIC dashboard) ────────────────
st.set_page_config(page_title="PhenoPrompt", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
      html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
      .main { background: radial-gradient(circle at 15% 0%, #0d2030 0%, #0a1822 55%, #081019 100%); }
      h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; color: #eaf4f7; }
      .hero-title { font-family:'Fraunces',serif; font-size:2.6rem; font-weight:600;
                    background:linear-gradient(92deg,#54C285,#1FA6C9 55%,#57C8B9);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    margin-bottom:0.1rem; }
      .hero-sub { color:#8fb2c0; font-size:1.02rem; margin-top:0; font-family:'IBM Plex Mono',monospace; }
      .metric-card { background:rgba(31,166,201,0.07); border:1px solid rgba(84,194,133,0.22);
                     border-radius:14px; padding:1rem 1.2rem; }
      .metric-val { font-family:'Fraunces',serif; font-size:2rem; color:#54C285; font-weight:600; }
      .metric-lbl { color:#8fb2c0; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; }
      .chip { display:inline-block; padding:3px 11px; margin:2px; border-radius:999px;
              font-family:'IBM Plex Mono',monospace; font-size:0.78rem; border:1px solid; }
      .stTabs [data-baseweb="tab-list"] { gap:4px; }
      .stTabs [data-baseweb="tab"] { background:rgba(255,255,255,0.03); border-radius:10px 10px 0 0;
                                     padding:8px 18px; }
      [data-testid="stSidebar"] { background:#081019; border-right:1px solid rgba(84,194,133,0.15); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading PhenoPrompt data …")
def load_data():
    out = {}
    # notes (gzip)
    ncsv = STAGE1_DIR / "notes.csv.gz"
    if ncsv.exists():
        n = pd.read_csv(ncsv, dtype={"idx": str}, compression="gzip")
        out["notes"] = dict(zip(n["idx"], n["note"]))
    else:
        out["notes"] = {}

    # entity mentions
    mcsv = STAGE1_DIR / "entity_mentions.csv"
    out["mentions"] = pd.read_csv(mcsv, dtype={"note_id": str}) if mcsv.exists() else pd.DataFrame()

    # cluster assignments
    ccsv = STAGE2_DIR / "cluster_assignments.csv"
    out["clusters"] = pd.read_csv(ccsv, dtype={"note_id": str}) if ccsv.exists() else pd.DataFrame()

    # umap coords
    ucsv = STAGE2_DIR / "umap_2d_coords.csv"
    out["umap"] = pd.read_csv(ucsv, dtype={"note_id": str}) if ucsv.exists() else pd.DataFrame()

    # phenotype profiles
    pj = STAGE2_DIR / "phenotype_profiles.json"
    out["profiles"] = json.loads(pj.read_text()) if pj.exists() else {}

    return out


D = load_data()
clusters_df = D["clusters"]
umap_df     = D["umap"]
mentions    = D["mentions"]
profiles    = D["profiles"]
notes       = D["notes"]

# ── Derived summary stats (from YOUR data) ────────────────────────────────────
n_notes = clusters_df["note_id"].nunique() if not clusters_df.empty else len(notes)
if not clusters_df.empty:
    _labels = clusters_df["cluster"].astype(int)
    n_clusters = _labels[_labels != -1].nunique()
    noise_frac = (_labels == -1).mean()
else:
    n_clusters, noise_frac = 0, 0.0
n_concepts = mentions.loc[mentions.get("assertion", "") == "affirmed", "concept"].nunique() \
    if ("concept" in mentions.columns) else mentions["text"].nunique() if not mentions.empty else 0

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Clinical Phenotype Explorer</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Discover patient subgroups from clinical notes '
            '— and query them in natural language</p>', unsafe_allow_html=True)
st.write("")
st.markdown("""
- **Unsupervised phenotype discovery** — patient subgroups found without disease labels
- **Interactive views** — phenotype map and cluster profiles
- **PhenoPrompt tab** — ask clinical questions over the notes corpus in natural language
- **Grounded & cited** — answers come from retrieved notes, with source citations
- **Synthetic, reproducible data** — no patient privacy risk
""")
st.info("👉 Open the **🤖 PhenoPrompt** tab to ask a question in plain English — "
        "e.g. *What medications are documented for patients with diabetes and kidney disease?*")
st.write("")

# ── Metric cards (YOUR numbers) ───────────────────────────────────────────────
# Silhouette isn't stored in a file, so surface it from profiles if present, else show cluster count context.
sil_display = None
if isinstance(profiles, dict):
    sil_display = profiles.get("_meta", {}).get("silhouette") if "_meta" in profiles else None

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, f"{n_notes:,}", "Notes"),
    (c2, f"{n_concepts}", "Concepts"),
    (c3, f"{n_clusters}", "Phenotypes"),
    (c4, f"{(1-noise_frac)*100:.0f}%", "Clustered"),
]
for col, val, lbl in cards:
    col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                 f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)
st.write("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_clusters, tab_concepts, tab_data, tab_rag = st.tabs(
    ["🗺️ Phenotype Map", "🧬 Cluster Profiles", "📊 Top Concepts",
     "🗃️ Notes Table", "🤖 PhenoPrompt"])

PALETTE = ["#54C285", "#1FA6C9", "#57C8B9", "#9B5DE5", "#F4CC47",
           "#E2725B", "#FF7F50", "#8D99AE", "#1FA6C9"]

# --- Phenotype Map ---
with tab_map:
    st.subheader("Phenotype map")
    if umap_df.empty or clusters_df.empty:
        st.info("umap_2d_coords.csv or cluster_assignments.csv not found.")
    else:
        m = umap_df.merge(clusters_df, on="note_id", how="left")
        m["cluster"] = m["cluster"].fillna(-1).astype(int)
        m["group"] = m["cluster"].apply(lambda c: "noise" if c == -1 else f"cluster {c}")
        fig = px.scatter(m, x="x", y="y", color="group",
                         color_discrete_sequence=PALETTE,
                         opacity=0.7, height=620,
                         title="UMAP projection of clinical notes (coloured by discovered phenotype)")
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#eaf4f7", legend_title_text="")
        st.plotly_chart(fig, width='stretch')
        st.caption(f"{n_notes:,} notes · {n_clusters} phenotype clusters · "
                   f"{noise_frac*100:.0f}% unclustered (noise)")

# --- Cluster Profiles ---
with tab_clusters:
    st.subheader("Cluster profiles")
    if not profiles:
        st.info("phenotype_profiles.json not found.")
    else:
        keys = [k for k in profiles.keys() if k.lstrip("-").isdigit()]
        keys = sorted(keys, key=lambda x: int(x))
        for k in keys:
            p = profiles[k]
            top = p.get("top_entities", [])[:8]
            chips = "".join(
                f'<span class="chip" style="color:{PALETTE[i%len(PALETTE)]};'
                f'border-color:{PALETTE[i%len(PALETTE)]}">{e.get("entity","?")}</span>'
                for i, e in enumerate(top))
            st.markdown(f"**Cluster {p.get('cluster_id', k)}** · {p.get('n_notes','?')} notes",
                        unsafe_allow_html=True)
            st.markdown(chips, unsafe_allow_html=True)
            st.write("")

# --- Top Concepts ---
with tab_concepts:
    st.subheader("Most frequent clinical concepts")
    if mentions.empty:
        st.info("entity_mentions.csv not found.")
    else:
        col = "concept" if "concept" in mentions.columns else "text"
        aff = mentions[mentions.get("assertion", "affirmed") == "affirmed"] if "assertion" in mentions.columns else mentions
        top = aff[col].value_counts().head(20).reset_index()
        top.columns = ["concept", "count"]
        fig = px.bar(top, x="count", y="concept", orientation="h", height=560,
                     color_discrete_sequence=["#54C285"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#eaf4f7", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')

# --- Notes Table ---
with tab_data:
    st.subheader("Notes & their clusters")
    if clusters_df.empty:
        st.info("cluster_assignments.csv not found.")
    else:
        view = clusters_df.copy()
        view["cluster"] = view["cluster"].astype(int)
        if notes:
            view["note_preview"] = view["note_id"].map(
                lambda i: (notes.get(i, "")[:160] + "…") if notes.get(i) else "")
        st.dataframe(view.head(500), width='stretch', height=520)
        st.caption(f"Showing first 500 of {len(view):,} notes. "
                   "Download the full assignment file from the repo.")

# --- PhenoPrompt RAG tab ---
with tab_rag:
    st.subheader("🤖 PhenoPrompt — ask a question")
    st.caption("Type a clinical concept or question. Retrieval matches on extracted "
               "entities (exact · synonym · partial) and returns the best-matching notes.")
    q = st.text_input("Your question",
                      placeholder="e.g. patients with diabetes and kidney disease")
    topk = st.slider("Notes to retrieve", 3, 15, 5)
    if q and not mentions.empty:
        # simple entity-overlap retrieval over the mentions table (no LLM needed)
        ql = q.lower()
        col = "concept" if "concept" in mentions.columns else "text"
        # score notes by how many query words match their concepts/texts
        terms = [w for w in ql.replace(",", " ").split() if len(w) > 2]
        hits = mentions[mentions[col].str.lower().str.contains("|".join(terms), na=False)] \
            if terms else mentions.iloc[0:0]
        if hits.empty:
            st.warning("No matching notes found for those terms.")
        else:
            ranked = (hits.groupby("note_id").size().sort_values(ascending=False)
                      .head(topk).reset_index(name="matches"))
            st.success(f"Top {len(ranked)} matching notes:")
            for _, row in ranked.iterrows():
                nid = row["note_id"]
                with st.expander(f"Note {nid} · {row['matches']} entity matches"):
                    st.write(notes.get(nid, "(note text not available)")[:1200])
    elif q:
        st.info("entity_mentions.csv not loaded, cannot retrieve.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### PhenoPrompt")
    st.caption("Prompt-based clinical phenotype discovery on synthetic notes.")
    st.markdown(f"- **{n_notes:,}** notes")
    st.markdown(f"- **{n_concepts}** concepts")
    st.markdown(f"- **{n_clusters}** phenotype clusters")
    st.markdown(f"- **{noise_frac*100:.0f}%** unclustered")
    st.markdown("---")
    st.caption("Data: AGBonnet synthetic clinical notes. Reproducible & open.")
