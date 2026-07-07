# PhenoPrompt — Sample Questions

Example natural-language questions you can ask in the **PhenoPrompt** tab of the
app. These are the queries used in the paper (IANLP 2026). Type them in either
**Standard** or **Agentic** retrieval mode.

Live app: https://phenoprompt-skausar-aivancity.streamlit.app/

---

## Simple questions

- Show me patients with heart failure and diabetes.
- What medications are documented for patients with diabetes?
- What symptoms are noted in patients with pneumonia?
- Describe patients with heart failure.
- What medications are documented for patients with diabetes and kidney disease?
- Describe patients with heart failure and fluid overload.
- What findings appear in patients with hypertension and diabetes?
- Which patients are breathless?
- Show me patients with CKD on diuretics.

## Complex, multi-condition questions

- Patients with heart failure, diabetes, and renal impairment on loop diuretics.
- Show me patients with type 2 diabetes, hypertension, and chronic kidney disease who are on an ACE inhibitor.
- Find patients with heart failure, atrial fibrillation, and shortness of breath who are taking a diuretic and an anticoagulant.
- Which patients have COPD with a productive cough and fever, and were started on a fluoroquinolone?
- Show me elderly patients with osteoporosis, a recent fracture, and joint pain who are also on an NSAID.
- Find patients with depression and anxiety who report fatigue and weight change, currently on an SSRI.
- Show me patients with sepsis, fever, and confusion who underwent a blood test and CT scan.
- Which diabetic patients on metformin and a statin also have hyperlipidemia and obesity?
- Find patients with chest pain and palpitations who had an ECG and echocardiogram, with a history of coronary artery disease.
- Show me T2DM patients with CKD and renal complications, on insulin, presenting with edema and breathless on exertion.
- Find patients with pneumonia and a respiratory infection, on amoxicillin, who also report dyspnoea and underwent a chest X-ray.

## Synonym-robustness examples

These single-concept queries demonstrate the synonym-expansion layer (each maps
to a canonical clinical concept during retrieval):

- type 2 diabetes  →  diabetes
- breathless  →  shortness of breath
- CKD  →  chronic kidney disease
- fluid overload  →  edema

## Agentic-mode examples (messy / misspelled input)

In **Agentic** mode, the tool-calling agent first normalises the query before
retrieval, so informal or misspelled questions still work, e.g.:

- diabetic patience deatils  →  (agent searches) diabetes
- ppl who cant breathe  →  shortness of breath
