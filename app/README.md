---
title: ASRS Triage Copilot
emoji: ✈
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Triage and similar-incident search for NASA ASRS narratives
---

# ASRS Triage Copilot

Semantic triage and similar-incident retrieval for NASA ASRS aviation safety narratives (MAAI 902: NLP and LLMs).

Paste a narrative to get suggested anomaly groups (fine-tuned DistilBERT), its recurring theme (BERTopic) and the most similar past reports (MiniLM + FAISS).

**Decision support only.** The model was trained on voluntary, unverified ASRS reports and has not been validated for operational use. A safety analyst must review every result.

Data: NASA Aviation Safety Reporting System (ASRS) public database, 2025 to September 2026. Report excerpts are shown for illustration; see the ASRS terms of use.
