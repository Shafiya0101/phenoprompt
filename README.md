# 🔎 PhenoPrompt

> **Prompt-based clinical phenotype discovery.** Build a phenotype space from clinical notes,
> then query it in natural language — type a clinical concept and retrieve the patient clusters
> where it is enriched, with no disease-specific algorithm required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shafiya0101/phenoprompt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Extends an earlier course project on structured-data phenotyping,
> [mimic-phenotyping](https://github.com/Shafiya0101/mimic-phenotyping), from a structured ICU
> table to unstructured clinical text — and adds the prompt-based query layer.

---

## What this is

A three-stage pipeline that turns raw clinical notes into a *queryable* map of patient
phenotypes:

```
Stage 1  medkit NER          notes ──▶ clinical entities (disorders, meds, findings)
Stage 2  embed + cluster     entity profiles ──▶ UMAP map + HDBSCAN phenotype clusters
Stage 3  prompt query (RAG)  free-text prompt ──▶ ranked, interpretable phenotype reports
```

The contribution is **Stage 3**: an entity-augmented retrieval layer (inspired by CLEAR,
López et al., *npj Digital Medicine* 2025) that scores each cluster by how **enriched** the
query's clinical entities are within it — prevalence × lift, so common entities can't pull in
unrelated clusters — then synthesises a readable phenotype report. Any condition is queryable
from a single index built **without** disease-specific specification, in contrast to the
"one algorithm per disease" norm in LLM phenotyping.

## Repository layout

```
notebooks/
  stage1_medkit_ner.ipynb            entity extraction (Colab + Google Drive)
  stage2_embedding_clustering.ipynb  LSA embedding, UMAP, HDBSCAN, phenotype mixes
  stage3_prompt_rag.ipynb            prompt-based query interface
phenoprompt_app.py                   Streamlit query app
data/phenoprompt/
  stage1_outputs/   commit: entity_count_matrix.csv, entity_mentions.csv, notes.csv
  stage2_outputs/   commit: phenotype_profiles.json, cluster_assignments.csv, umap_2d_coords.csv
requirements.txt                     app dependencies
```

## Running the notebooks (Colab)

The notebooks read and write a shared Google Drive folder so results survive session
disconnects. Run them **in order**:

1. `stage1_medkit_ner.ipynb` — downloads the corpus, runs medkit, writes
   `MyDrive/phenoprompt/stage1_outputs/`. (Slowest stage.)
2. `stage2_embedding_clustering.ipynb` — clusters and writes
   `MyDrive/phenoprompt/stage2_outputs/` (including `phenotype_profiles.json`).
3. `stage3_prompt_rag.ipynb` — loads the above and answers prompts. Defaults to a no-API
   "fallback" report writer; set `LLM_BACKEND` to `anthropic`/`openai` for LLM-written
   cluster labels and narratives.

## Running the app

```
pip install -r requirements.txt
streamlit run phenoprompt_app.py
```

The app loads the committed files under `data/phenoprompt/`. Download the six output files
from your Drive into those folders (see the `README.txt` in each) before deploying to
Streamlit Cloud, which serves from the GitHub repo rather than from Drive.

## Data

Runs on the **synthetic** `AGBonnet/augmented-clinical-notes` corpus (HuggingFace) — no
PhysioNet credentialing or PHI restriction. Porting to real MIMIC-III/IV discharge summaries
is planned and would require credentialed access.

## Status & known limitations

The pipeline runs end to end. On a 500-note synthetic sample the current clustering is
**preliminary**: 6 clusters, ~70% of notes assigned to noise, silhouette ≈ 0.27, DBCV ≈ 0.04.
Planned improvements: the full ~30k-note corpus; a UMLS-grounded NER (QuickUMLS / scispaCy) to
expand the entity vocabulary; medkit-on-query for extraction parity; an under-coded-conditions
analysis against structured ICD codes; and a formal evaluation (NER F1, cluster validity,
query precision/recall).

## License

MIT — see [LICENSE](LICENSE).
