"""
PhenoPrompt — Clinical RAG (Streamlit app).

Ask a clinical question; the app retrieves the most relevant notes (entity-augmented) and
Mistral writes a grounded answer that cites the source notes. Reads the committed Stage 1/2
outputs under data/phenoprompt/.

Deploy on Streamlit Cloud:
  - main file: rag_app.py
  - add MISTRAL_API_KEY in the app's Settings -> Secrets
Run locally:
  export MISTRAL_API_KEY=...   &&   streamlit run rag_app.py
"""
import os, re, math, json, textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR   = Path(__file__).parent / "data" / "phenoprompt"
STAGE1_DIR = DATA_DIR / "stage1_outputs"
STAGE2_DIR = DATA_DIR / "stage2_outputs"

CHAT_MODEL = "mistral-small-latest"
TOP_K_DEFAULT = 5

SYNONYMS = {
    "type 2 diabetes": ["diabetes"], "t2dm": ["diabetes"], "diabetic": ["diabetes"],
    "renal": ["kidney", "renal failure", "chronic kidney disease"],
    "kidney": ["renal failure", "chronic kidney disease"], "ckd": ["chronic kidney disease"],
    "hf": ["heart failure"], "chf": ["heart failure"], "sob": ["shortness of breath"],
    "breathless": ["shortness of breath"], "fluid overload": ["edema"], "swelling": ["edema"],
    "diuretic": ["furosemide"], "lung infection": ["pneumonia"], "chest infection": ["pneumonia"],
    "respiratory infection": ["pneumonia"], "infection": ["pneumonia"],
}

SYSTEM = (
    "You are a careful clinical informatics assistant answering questions about a cohort of "
    "patient notes. Use ONLY the numbered notes provided as context. Cite the notes you use "
    "with their id in square brackets like [note 1234]. If the notes do not contain enough "
    "information to answer, say so plainly. Do NOT invent diagnoses, drugs, values, or guidance."
)


def get_key():
    k = os.environ.get("MISTRAL_API_KEY")
    if k:
        return k
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except Exception:
        return None


@st.cache_resource
def load_index():
    # notes
    notes = {}
    ncsv = STAGE1_DIR / "notes.csv.gz"
    if ncsv.exists():
        n = pd.read_csv(ncsv, dtype={"idx": str}, compression="gzip")
        notes = dict(zip(n["idx"], n["note"]))
    # entity mentions (affirmed) -> per-note entities + idf
    note_ents, idf, vocab = {}, {}, set()
    mcsv = STAGE1_DIR / "entity_mentions.csv"
    if mcsv.exists() and notes:
        m = pd.read_csv(mcsv, dtype={"note_id": str})
        m = m[(m["assertion"] == "affirmed") & (m["note_id"].isin(notes))]
        g = m.groupby("note_id")["text"].apply(list)
        note_ents = {nid: g.get(nid, []) for nid in notes}
        dfc = {}
        for ents in note_ents.values():
            for e in set(ents):
                dfc[e] = dfc.get(e, 0) + 1
        N = max(len(notes), 1)
        idf = {e: math.log((N + 1) / (c + 1)) + 1 for e, c in dfc.items()}
        vocab = set(idf)
    # cluster assignments (optional)
    note2cluster = {}
    ccsv = STAGE2_DIR / "cluster_assignments.csv"
    if ccsv.exists():
        c = pd.read_csv(ccsv, dtype={"note_id": str})
        note2cluster = dict(zip(c["note_id"], c["cluster"]))
    return notes, note_ents, idf, vocab, note2cluster


def query_entities(q, vocab):
    q = q.lower(); hits = {}
    for e in vocab:
        if e in q:
            hits[e] = max(hits.get(e, 0.0), 1.0)
    for syn, tgts in SYNONYMS.items():
        if syn in q:
            for t in tgts:
                if t in vocab:
                    hits[t] = max(hits.get(t, 0.0), 0.9)
    qtokens = [w for w in re.findall(r"[a-z]+", q) if len(w) >= 4]
    for e in vocab:
        if set(e.split()) & set(qtokens):
            hits[e] = max(hits.get(e, 0.0), 0.6)
    return hits


def retrieve(question, notes, note_ents, idf, vocab, k):
    qe = query_entities(question, vocab)
    if not qe:
        return []
    scored = []
    for nid, ents in note_ents.items():
        if not ents:
            continue
        tf = {}
        for e in ents:
            tf[e] = tf.get(e, 0) + 1
        s = sum(w * tf.get(e, 0) * idf.get(e, 1.0) for e, w in qe.items())
        if s > 0:
            scored.append((nid, round(float(s), 3)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def answer(question, notes, hits, note2cluster, key):
    blocks = []
    for nid, _ in hits:
        cl = note2cluster.get(nid)
        tag = f" (phenotype cluster {cl})" if cl not in (None, -1) else ""
        blocks.append(f"[note {nid}]{tag}\n{notes.get(nid,'')[:1200]}")
    context = "\n\n".join(blocks)
    if not key:
        return ("**No MISTRAL_API_KEY configured** — showing retrieved evidence instead of an "
                "LLM answer.\n\n" + "\n\n".join(
                    f"[note {nid}] (score {s})\n\n{notes.get(nid,'')[:400]}" for nid, s in hits))
    from mistralai import Mistral
    cl = Mistral(api_key=key)
    resp = cl.chat.complete(model=CHAT_MODEL, temperature=0.1, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context notes:\n\n{context}\n\nQuestion: {question}"},
    ])
    return resp.choices[0].message.content


SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_clinical_notes",
        "description": ("Search the clinical-note corpus for notes relevant to a clinical "
                        "concept (condition, medication, symptom, or finding). Returns matching "
                        "patient notes. Call this to ground your answer in real notes before "
                        "responding."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": ("A concise clinical search phrase capturing the condition(s), "
                                    "medication(s), or symptom(s) to look for, e.g. "
                                    "'diabetes kidney disease'. Correct spelling and extract the "
                                    "clinical concepts from the user's wording."),
                }
            },
            "required": ["query"],
        },
    },
}]


def agentic_answer(question, notes, note_ents, idf, vocab, note2cluster, key, k):
    """One-step agentic RAG: the model reformulates the question, calls the search tool,
    then answers from the retrieved notes. Falls back to standard retrieval on any issue."""
    from mistralai import Mistral
    client = Mistral(api_key=key)
    messages = [
        {"role": "system", "content": SYSTEM + " You have a tool to search the clinical notes. "
         "Always call it before answering, normalising the user's wording into clean clinical "
         "terms (fix spelling, extract the conditions/medications)."},
        {"role": "user", "content": question},
    ]
    # step 1 — force a single tool call; the model decides the search query
    r1 = client.chat.complete(model=CHAT_MODEL, messages=messages,
                              tools=SEARCH_TOOL, tool_choice="any", temperature=0.1)
    msg = r1.choices[0].message
    if not getattr(msg, "tool_calls", None):
        hits = retrieve(question, notes, note_ents, idf, vocab, k)
        return {"answer": answer(question, notes, hits, note2cluster, key),
                "hits": hits, "search_query": question, "agentic": False}
    tc = msg.tool_calls[0]
    try:
        search_query = json.loads(tc.function.arguments)["query"]
    except Exception:
        search_query = question
    # step 2 — run the retrieval tool with the model's cleaned query
    hits = retrieve(search_query, notes, note_ents, idf, vocab, k)
    tool_content = "\n\n".join(f"[note {nid}]\n{notes.get(nid,'')[:1200]}"
                               for nid, _ in hits) or "No notes found."
    # step 3 — return the tool result to the model for the final grounded answer
    messages.append(msg)
    messages.append({"role": "tool", "name": tc.function.name,
                     "content": tool_content, "tool_call_id": tc.id})
    r2 = client.chat.complete(model=CHAT_MODEL, messages=messages, temperature=0.1)
    return {"answer": r2.choices[0].message.content, "hits": hits,
            "search_query": search_query, "agentic": True}


@st.cache_resource
def load_clustering():
    """Loads Stage 2 outputs for the dashboard. Returns (profiles, map_df, labels, sizes)
    or (None, ...) if unavailable — the RAG tab works regardless."""
    pf = STAGE2_DIR / "phenotype_profiles.json"
    ca = STAGE2_DIR / "cluster_assignments.csv"
    um = STAGE2_DIR / "umap_2d_coords.csv"
    if not (pf.exists() and ca.exists()):
        return None, None, {}, {}
    profiles = json.loads(pf.read_text())
    assign = pd.read_csv(ca, dtype={"note_id": str})

    # cluster -> short label from its top two entities
    labels = {}
    for cid, p in profiles.items():
        ents = [e.get("entity", "") for e in p.get("top_entities", [])[:2]]
        labels[int(cid)] = " / ".join([e for e in ents if e]) or f"cluster {cid}"

    # build the map dataframe (x, y, cluster, note_id) with robust column detection
    map_df = None
    if um.exists():
        u = pd.read_csv(um)
        cols = list(u.columns)
        nid = next((c for c in cols if "id" in c.lower()), None)
        coord = [c for c in cols if c != nid and "cluster" not in c.lower()
                 and pd.api.types.is_numeric_dtype(u[c])][:2]
        if len(coord) == 2:
            map_df = pd.DataFrame({"x": u[coord[0]].values, "y": u[coord[1]].values})
            if nid:
                map_df["note_id"] = u[nid].astype(str).values
            if nid and "note_id" in assign.columns:
                map_df = map_df.merge(assign[["note_id", "cluster"]], on="note_id", how="left")
            elif len(map_df) == len(assign):
                map_df["cluster"] = assign["cluster"].values
            map_df["cluster"] = map_df.get("cluster", pd.Series([-1] * len(map_df))).fillna(-1).astype(int)
            map_df["phenotype"] = map_df["cluster"].map(
                lambda c: "noise" if c == -1 else f"C{c}: {labels.get(c, '')}")

    sizes = assign["cluster"].value_counts().sort_index().to_dict()
    return profiles, map_df, labels, sizes


def cluster_profile_df(profiles, cid, top=12):
    rows = profiles[str(cid)].get("top_entities", [])[:top]
    df = pd.DataFrame(rows)
    keep = [c for c in ["entity", "score", "prevalence_cluster"] if c in df.columns]
    return df[keep] if keep else df


# ----------------------------------------------------------------------------
st.set_page_config(page_title="PhenoPrompt", page_icon="🩺", layout="wide")
st.title("PhenoPrompt")
st.caption("Prompt-based clinical phenotype discovery — query a phenotype space built from "
           "clinical notes, and explore the discovered clusters.")

notes, note_ents, idf, vocab, note2cluster = load_index()
key = get_key()
try:
    profiles, map_df, clabels, sizes = load_clustering()
except Exception as e:
    profiles, map_df, clabels, sizes = None, None, {}, {}

# ---- metric cards (top) ----
n_clusters = len(profiles) if profiles else 0
noise_pct = (100 * sizes.get(-1, 0) / sum(sizes.values())) if sizes else 0.0
m = st.columns(4)
m[0].metric("Notes", f"{len(notes):,}")
m[1].metric("Entities", f"{len(vocab):,}")
m[2].metric("Phenotypes", n_clusters)
m[3].metric("Noise", f"{noise_pct:.0f}%")

tab_rag, tab_map, tab_prof, tab_feat, tab_dq, tab_cohort = st.tabs(
    ["🩺 Clinical RAG", "🗺️ Phenotype Map", "📊 Cluster Profiles",
     "🔬 Feature Drivers", "✅ Data Quality", "📋 Cohort Table"])

# ============================ TAB 1 — Clinical RAG ============================
with tab_rag:
    if not notes:
        st.error("No notes found under data/phenoprompt/stage1_outputs/. "
                 "Commit notes.csv and entity_mentions.csv to the repo.")
    else:
        c1, c2 = st.columns([3, 1])
        with c2:
            mode = st.radio("Mode", ["Standard RAG", "Agentic RAG (tool-calling)"], index=0,
                            help="Agentic mode lets the model reformulate your question into a "
                                 "clean clinical search before retrieving.")
            top_k = st.slider("Notes to retrieve", 1, 10, TOP_K_DEFAULT)
            st.caption(f"Mistral key: {'set ✓' if key else 'missing ✗'}")
        with c1:
            question = st.text_input(
                "Clinical question",
                "What medications are documented for patients with diabetes and kidney disease?")
            ask = st.button("Ask", type="primary")

        if ask and question:
            agentic_mode = mode.startswith("Agentic") and key
            if agentic_mode:
                with st.spinner("Agent reformulating, searching, and answering..."):
                    try:
                        res = agentic_answer(question, notes, note_ents, idf, vocab,
                                             note2cluster, key, top_k)
                        hits = res["hits"]; resp = res["answer"]
                        if res.get("agentic"):
                            st.info(f"🔧 The agent searched for: **{res['search_query']}**")
                    except Exception as e:
                        st.warning(f"Agentic mode failed ({e}); using standard RAG.")
                        agentic_mode = False
            if not agentic_mode:
                hits = retrieve(question, notes, note_ents, idf, vocab, top_k)
                if hits:
                    with st.spinner("Retrieving notes and generating a grounded answer..."):
                        resp = answer(question, notes, hits, note2cluster, key)
            if not hits:
                st.warning("No relevant notes retrieved. Try rephrasing or different terms.")
            else:
                st.subheader("Answer"); st.markdown(resp)
                clusters = sorted({note2cluster.get(n) for n, _ in hits
                                   if note2cluster.get(n) not in (None, -1)})
                cc = st.columns(2)
                cc[0].write("**Source notes:** " + ", ".join(str(n) for n, _ in hits))
                if clusters:
                    cc[1].write("**Phenotype clusters:** " + ", ".join(map(str, clusters)))
                st.subheader("Retrieved evidence")
                for nid, score in hits:
                    cl = note2cluster.get(nid)
                    lbl = f"note {nid}  ·  score {score}" + (f"  ·  cluster {cl}"
                                                             if cl not in (None, -1) else "")
                    with st.expander(lbl):
                        st.write(notes.get(nid, ""))

# guard the dashboard tabs when clustering data is missing
_no_clusters = profiles is None
_dash_msg = ("Clustering outputs not found under data/phenoprompt/stage2_outputs/. "
             "Commit phenotype_profiles.json, cluster_assignments.csv, and umap_2d_coords.csv "
             "to enable the dashboard.")

# ============================ TAB 2 — Phenotype Map ==========================
with tab_map:
    if _no_clusters or map_df is None:
        st.info(_dash_msg if _no_clusters else "umap_2d_coords.csv not found or unrecognised.")
    else:
        st.subheader("UMAP embedding coloured by phenotype")
        try:
            import plotly.express as px
            order = sorted(map_df["phenotype"].unique(),
                           key=lambda s: (s == "noise", s))
            fig = px.scatter(map_df, x="x", y="y", color="phenotype",
                             category_orders={"phenotype": order},
                             hover_data=[c for c in ["note_id", "cluster"] if c in map_df.columns],
                             opacity=0.75, height=560)
            fig.update_traces(marker=dict(size=6))
            fig.update_layout(legend_title_text="phenotype", xaxis_title="UMAP-1",
                              yaxis_title="UMAP-2")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.scatter_chart(map_df, x="x", y="y", color="phenotype", height=520)
        st.caption("Each point is a note, positioned by entity-profile similarity and coloured "
                   "by its discovered phenotype cluster (grey = noise / unclustered).")

# ============================ TAB 3 — Cluster Profiles =======================
with tab_prof:
    if _no_clusters:
        st.info(_dash_msg)
    else:
        ids = sorted(int(c) for c in profiles)
        choice = st.selectbox("Cluster", ids,
                              format_func=lambda c: f"Cluster {c} — {clabels.get(c,'')}")
        prof = profiles[str(choice)]
        st.markdown(f"**Cluster {choice} — {clabels.get(choice,'')}**  ·  "
                    f"{prof.get('n_notes','?')} notes")
        dfp = cluster_profile_df(profiles, choice)
        if "score" in dfp.columns:
            st.bar_chart(dfp.set_index("entity")["score"], height=340)
        st.dataframe(dfp, hide_index=True, use_container_width=True)

# ============================ TAB 4 — Feature Drivers ========================
with tab_feat:
    if _no_clusters:
        st.info(_dash_msg)
    else:
        st.caption("Entities most characteristic of each phenotype (by within-cluster "
                   "prevalence). These are the features that 'drive' each cluster.")
        ids = sorted(int(c) for c in profiles)
        sel = st.selectbox("Cluster ", ids, key="feat",
                           format_func=lambda c: f"Cluster {c} — {clabels.get(c,'')}")
        dfp = cluster_profile_df(profiles, sel, top=10)
        ycol = "prevalence_cluster" if "prevalence_cluster" in dfp.columns else (
               "score" if "score" in dfp.columns else None)
        if ycol:
            st.bar_chart(dfp.set_index("entity")[ycol], height=360)
        st.dataframe(dfp, hide_index=True, use_container_width=True)

# ============================ TAB 5 — Data Quality ===========================
with tab_dq:
    if _no_clusters:
        st.info(_dash_msg)
    else:
        total = sum(sizes.values()); noise = sizes.get(-1, 0)
        q = st.columns(4)
        q[0].metric("Notes clustered", f"{total - noise:,}")
        q[1].metric("Noise notes", f"{noise:,}")
        q[2].metric("Clusters", n_clusters)
        q[3].metric("Silhouette (Stage 2)", "≈0.27")
        st.caption("Silhouette ≈ 0.27 and DBCV ≈ 0.04 were reported by Stage 2; the high noise "
                   "fraction reflects the small synthetic development sample and the rule-based "
                   "entity vocabulary. Improving these is the main item of future work.")
        st.subheader("Cluster sizes (incl. noise)")
        size_df = (pd.DataFrame({"cluster": list(sizes.keys()), "notes": list(sizes.values())})
                   .replace({"cluster": {-1: "noise"}}).set_index("cluster"))
        st.bar_chart(size_df, height=320)

# ============================ TAB 6 — Cohort Table ===========================
with tab_cohort:
    if _no_clusters:
        st.info(_dash_msg)
    else:
        total = sum(v for k, v in sizes.items() if k != -1) or 1
        rows = []
        for cid in sorted(int(c) for c in profiles):
            p = profiles[str(cid)]
            top5 = ", ".join(e.get("entity", "") for e in p.get("top_entities", [])[:5])
            n = p.get("n_notes", sizes.get(cid, 0))
            rows.append({"cluster": cid, "label": clabels.get(cid, ""), "notes": n,
                         "% of cohort": f"{100*n/total:.1f}%", "top entities": top5})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("One row per discovered phenotype cluster (noise excluded from the cohort %).")
