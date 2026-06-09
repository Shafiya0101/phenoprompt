"""
PhenoPrompt — Streamlit query interface.

Deploys the Stage 3 prompt-based phenotype query on top of the artifacts produced by
Stage 1 (medkit NER) and Stage 2 (embedding + clustering). Commit those output files into
the repo under  data/phenoprompt/  and Streamlit Cloud will serve this page.

Run locally:   streamlit run phenoprompt_app.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Where the committed Stage 1/2 outputs live (relative to the repo root).
# ----------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent / "data" / "phenoprompt"
STAGE1_DIR = DATA_DIR / "stage1_outputs"
STAGE2_DIR = DATA_DIR / "stage2_outputs"

TOP_K, ALPHA, MIN_SCORE = 3, 0.7, 1e-6

SYNONYMS = {
    "type 2 diabetes": ["diabetes"], "t2dm": ["diabetes"], "diabetic": ["diabetes"],
    "renal": ["chronic kidney disease", "renal failure", "kidney"],
    "kidney": ["chronic kidney disease", "renal failure"], "ckd": ["chronic kidney disease"],
    "hf": ["heart failure"], "chf": ["heart failure"], "cardiac failure": ["heart failure"],
    "fluid overload": ["edema"], "swelling": ["edema"], "sob": ["shortness of breath"],
    "breathless": ["shortness of breath"], "diuretic": ["furosemide"],
    "lung infection": ["pneumonia"], "chest infection": ["pneumonia"],
    "heart attack": ["myocardial infarction"], "high blood pressure": ["hypertension"],
}


class PhenoSpace:
    """Queryable index over Stage 2 phenotype clusters (enrichment-weighted retrieval)."""

    def __init__(self, stage1_dir, stage2_dir):
        s1, s2 = Path(stage1_dir), Path(stage2_dir)
        self.profiles = json.loads((s2 / "phenotype_profiles.json").read_text())
        self.assign   = pd.read_csv(s2 / "cluster_assignments.csv", dtype={"note_id": str})
        self.count    = pd.read_csv(s1 / "entity_count_matrix.csv", index_col=0)
        self.count.index = self.count.index.astype(str)
        self.vocab    = list(self.count.columns)
        self.mentions = pd.read_csv(s1 / "entity_mentions.csv", dtype={"note_id": str})
        nc = s1 / "notes.csv"
        self.note_texts = (dict(zip(pd.read_csv(nc, dtype={"idx": str})["idx"],
                                    pd.read_csv(nc, dtype={"idx": str})["note"]))
                           if nc.exists() else {})
        self.cluster_ids = sorted(int(c) for c in self.profiles)
        self.corpus_prev = (self.count > 0).mean()
        self.prev_vec = {}
        for cl in self.cluster_ids:
            ids = self.assign.loc[self.assign.cluster == cl, "note_id"].values
            self.prev_vec[cl] = (self.count.reindex(ids) > 0).mean().reindex(self.vocab).fillna(0).values

    def extract_query_entities(self, query):
        q = query.lower(); hits = {}
        for e in self.vocab:
            if e in q: hits[e] = max(hits.get(e, 0.0), 1.0)
        for syn, targets in SYNONYMS.items():
            if syn in q:
                for t in targets:
                    if t in self.vocab: hits[t] = max(hits.get(t, 0.0), 0.9)
        for tok in re.findall(r"[a-z]+", q):
            if len(tok) < 4: continue
            for e in self.vocab:
                if tok in e.split(): hits[e] = max(hits.get(e, 0.0), 0.7)
        return hits

    def retrieve(self, query, top_k=TOP_K, alpha=ALPHA):
        qents = self.extract_query_entities(query)
        if not qents: return [], qents
        vidx = {e: i for i, e in enumerate(self.vocab)}
        cp   = self.corpus_prev.reindex(self.vocab).fillna(0).values
        qvec = np.array([qents.get(e, 0.0) for e in self.vocab], dtype=float)
        qn   = qvec / (np.linalg.norm(qvec) + 1e-8)
        ent_scores, cosines, rows = [], [], []
        for cl in self.cluster_ids:
            pv = self.prev_vec[cl]; ent = 0.0
            for e, w in qents.items():
                if e in vidx:
                    i = vidx[e]; ent += w * pv[i] * (pv[i] / (cp[i] + 1e-8))
            pvn = pv / (np.linalg.norm(pv) + 1e-8)
            ent_scores.append(ent); cosines.append(float(qn @ pvn))
            rows.append(dict(cluster=cl, n_notes=self.profiles[str(cl)]["n_notes"],
                             matched_entities=[e for e in qents if e in vidx and pv[vidx[e]] > 0]))
        ent_scores, cosines = np.array(ent_scores), np.array(cosines)
        def mm(a):
            rng = a.max() - a.min()
            return (a - a.min()) / rng if rng > 1e-12 else np.zeros_like(a)
        hybrid = alpha * mm(ent_scores) + (1 - alpha) * mm(cosines)
        for r, h, e, c in zip(rows, hybrid, ent_scores, cosines):
            r.update(hybrid=round(float(h), 4), entity_score=round(float(e), 4),
                     cosine=round(float(c), 4))
        rows = [r for r in rows if r["entity_score"] > MIN_SCORE]
        rows.sort(key=lambda r: r["hybrid"], reverse=True)
        return rows[:top_k], qents

    def phenotype_mix(self, cluster_id, top=10):
        return pd.DataFrame(self.profiles[str(cluster_id)]["top_entities"]).head(top)

    def fragments(self, cluster_id, query_entities, n=2):
        ids = set(self.assign.loc[self.assign.cluster == cluster_id, "note_id"])
        m = self.mentions[self.mentions.note_id.isin(ids)
                          & self.mentions.text.isin(list(query_entities))
                          & (self.mentions.assertion == "affirmed")]
        out = []
        for nid in list(dict.fromkeys(m.note_id))[:n]:
            out.append(dict(note_id=nid, text=self.note_texts.get(nid, "")))
        return out


@st.cache_resource
def load_space():
    return PhenoSpace(STAGE1_DIR, STAGE2_DIR)


def label_for(mix):
    top = mix["entity"].head(2).tolist()
    return " + ".join(t.title() for t in top) + " phenotype"


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="PhenoPrompt", page_icon="🔎", layout="wide")
st.title("PhenoPrompt — prompt-based phenotype query")
st.caption("Type a clinical concept; retrieve the phenotype clusters where it is enriched.")

try:
    ps = load_space()
except Exception as e:
    st.error(f"Could not load phenotype index from {DATA_DIR}. "
             f"Commit the Stage 1/2 output files there. ({e})")
    st.stop()

with st.sidebar:
    st.metric("Clusters", len(ps.cluster_ids))
    st.metric("Entities in vocabulary", len(ps.vocab))
    st.metric("Notes indexed", len(ps.note_texts))
    top_k = st.slider("Clusters to return", 1, min(6, len(ps.cluster_ids)), 3)

query = st.text_input("Clinical query",
                      "type 2 diabetes with renal complications")

if query:
    ranked, qents = ps.retrieve(query, top_k=top_k)
    st.write("**Extracted entities:** " + (", ".join(qents) if qents else "none"))

    if not ranked:
        st.warning("No clusters matched. Try terms present in the corpus, "
                   "or extend the synonym map.")
    else:
        for i, r in enumerate(ranked, 1):
            mix = ps.phenotype_mix(r["cluster"])
            with st.container(border=True):
                st.subheader(f"{i}. Cluster {r['cluster']} — {label_for(mix)}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Patients", r["n_notes"])
                c2.metric("Hybrid score", f"{r['hybrid']:.3f}")
                c3.write("**Matched:** " + (", ".join(r["matched_entities"]) or "—"))

                disp = mix.copy()
                disp["corpus_prevalence"] = disp["entity"].map(
                    lambda e: round(float(ps.corpus_prev.get(e, 0.0)), 3))
                st.dataframe(disp, hide_index=True, use_container_width=True)

                frags = ps.fragments(r["cluster"], qents)
                if frags and frags[0]["text"]:
                    with st.expander("Representative note fragment (provenance)"):
                        st.write(frags[0]["text"][:600])
