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
    ncsv = STAGE1_DIR / "notes.csv"
    if ncsv.exists():
        n = pd.read_csv(ncsv, dtype={"idx": str})
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


# ----------------------------------------------------------------------------
st.set_page_config(page_title="PhenoPrompt — Clinical RAG", page_icon="🩺", layout="wide")
st.title("PhenoPrompt — Clinical RAG")
st.caption("Ask a clinical question; answers are grounded in retrieved patient notes, with citations.")

notes, note_ents, idf, vocab, note2cluster = load_index()
key = get_key()

with st.sidebar:
    st.metric("Notes indexed", len(notes))
    st.metric("Distinct entities", len(vocab))
    st.metric("Mistral key", "set ✓" if key else "missing ✗")
    mode = st.radio("Mode", ["Standard RAG", "Agentic RAG (tool-calling)"], index=0,
                    help="Agentic mode lets the model reformulate your question into a clean "
                         "clinical search before retrieving — more robust to phrasing/typos.")
    top_k = st.slider("Notes to retrieve", 1, 10, TOP_K_DEFAULT)
    if not key:
        st.info("Add MISTRAL_API_KEY in Settings → Secrets to enable LLM answers. "
                "Without it, the app shows retrieved evidence only.")

if not notes:
    st.error("No notes found under data/phenoprompt/stage1_outputs/. "
             "Commit notes.csv and entity_mentions.csv to the repo.")
    st.stop()

question = st.text_input("Clinical question",
                         "What medications are documented for patients with diabetes and kidney disease?")

if st.button("Ask", type="primary") and question:
    agentic_mode = mode.startswith("Agentic") and key
    search_query = question
    if agentic_mode:
        with st.spinner("Agent is reformulating the query, searching notes, and answering..."):
            try:
                res = agentic_answer(question, notes, note_ents, idf, vocab,
                                     note2cluster, key, top_k)
                hits = res["hits"]; resp = res["answer"]; search_query = res["search_query"]
                if res.get("agentic"):
                    st.info(f"🔧 The agent searched the notes for: **{search_query}**")
            except Exception as e:
                st.warning(f"Agentic mode failed ({e}); falling back to standard RAG.")
                agentic_mode = False

    if not agentic_mode:
        hits = retrieve(question, notes, note_ents, idf, vocab, top_k)
        if not hits:
            st.warning("No relevant notes retrieved. Try different terms (the entity index is "
                       "vocabulary-bound) or rephrase.")
            st.stop()
        with st.spinner("Retrieving notes and generating a grounded answer..."):
            resp = answer(question, notes, hits, note2cluster, key)

    if not hits:
        st.warning("No relevant notes retrieved for the agent's search. Try rephrasing.")
    else:
        st.subheader("Answer")
        st.markdown(resp)

        clusters = sorted({note2cluster.get(n) for n, _ in hits
                           if note2cluster.get(n) not in (None, -1)})
        cols = st.columns(2)
        cols[0].write("**Source notes:** " + ", ".join(str(n) for n, _ in hits))
        if clusters:
            cols[1].write("**Phenotype clusters referenced:** " + ", ".join(map(str, clusters)))

        st.subheader("Retrieved evidence")
        for nid, score in hits:
            cl = note2cluster.get(nid)
            label = f"note {nid}  ·  score {score}" + (f"  ·  cluster {cl}"
                                                       if cl not in (None, -1) else "")
            with st.expander(label):
                st.write(notes.get(nid, ""))
