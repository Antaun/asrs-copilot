"""ASRS Triage Copilot: Hugging Face Space entry point (CPU).

Semantic triage and similar-incident retrieval for NASA ASRS safety narratives.
Everything the app loads sits in ./artifacts (written by the notebook's export cell).
"""

import html, inspect, json, random, re
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ART = Path(__file__).parent / "artifacts"
MAX_LEN = 512
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


# ---------- text preparation (same code as the notebook, so inputs match training) ----------
def normalise_placeholders(text):
    text = text.replace(";", ",")
    text = re.sub(r"\bZ{3,}\d*\b", "ZZZ", text)  # airports
    text = re.sub(r"\bXX+[LRC]?\b", "XX", text)  # runway numbers
    return re.sub(r"\s+", " ", text).strip()


ABBREV = {
    "ATC": "air traffic control",
    "FO": "first officer",
    "PF": "pilot flying",
    "PM": "pilot monitoring",
    "QRH": "quick reference handbook",
    "ILS": "instrument landing system",
    "TCAS": "traffic collision avoidance system",
    "VMC": "visual meteorological conditions",
    "IMC": "instrument meteorological conditions",
    "MEL": "minimum equipment list",
    "FBO": "fixed base operator",
    "MLG": "main landing gear",
    "RA": "resolution advisory",
    "TA": "traffic advisory",
    "VNAV": "vertical navigation",
    "FMS": "flight management system",
    "APU": "auxiliary power unit",
    "GPWS": "ground proximity warning system",
    "AGL": "above ground level",
    "MSL": "mean sea level",
    "IFR": "instrument flight rules",
    "VFR": "visual flight rules",
}
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(sorted(ABBREV, key=len, reverse=True)) + r")\b"
)


def expand_abbreviations(text):
    return _ABBREV_RE.sub(lambda m: ABBREV[m.group(1)], text)


def head_tail_ids(text, tokenizer, max_len=MAX_LEN):
    head = max_len // 4
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    room = max_len - 2
    if len(ids) > room:
        ids = ids[:head] + ids[-(room - head) :]
    return [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]


# ---------- embeddings and search (same logic as the notebook) ----------
SENT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text, tokenizer, max_tokens=200):
    sents, chunks, cur, cur_n = SENT_RE.split(text), [], [], 0
    for s in sents:
        n = len(tokenizer(s, add_special_tokens=False)["input_ids"])
        if cur and cur_n + n > max_tokens:
            chunks.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def embed_truncated(model, texts, batch_size=64):
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


def embed_chunked(model, texts, max_tokens=200, batch_size=64):
    chunks, owner = [], []
    for i, t in enumerate(texts):
        cs = chunk_text(t, model.tokenizer, max_tokens)
        chunks += cs
        owner += [i] * len(cs)
    ce = model.encode(
        chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    out = np.zeros((len(texts), ce.shape[1]), dtype="float32")
    np.add.at(out, np.array(owner), ce)
    return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype("float32")


def build_faiss_index(emb):
    index = faiss.IndexFlatIP(emb.shape[1])  # exact search; fine for ~6,000 vectors
    index.add(emb)
    return index


def search_similar(index, query_emb, k=5):
    return index.search(query_emb, k)


def sample_topic_docs(texts, topics, topic_id, n=3, seed=0, chars=300):
    idx = np.where(topics == topic_id)[0]
    pick = np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False)
    return [texts.iloc[i][:chars] for i in pick]


# ======================================================================================
# The demo (styling, compute, render and wiring functions)
# ======================================================================================
TITLE = "ASRS Copilot"
# palette shared with the project deck: dark values and light values
D = dict(
    page_a="#1B1E23",
    page_b="#2B2F36",
    panel="#2B2F36",
    chip="#343942",
    edge="#3A3F48",
    ink="#F2EFE8",
    soft="#D6D2C8",
    muted="#A6A296",
    accent_ink="#F5C518",
    input="#1B1E23",
    track="#3A3F48",
    off="#5F5B53",
)
L = dict(
    page_a="#FBF8F0",
    page_b="#F0EBDD",
    panel="#FFFFFF",
    chip="#F3EFE4",
    edge="#E3DDCE",
    ink="#1B1E23",
    soft="#46433B",
    muted="#6B665A",
    accent_ink="#8A5A00",
    input="#FBF9F4",
    track="#E8E2D2",
    off="#B9B3A3",
)
YELLOW, ORANGE = "#F5C518", "#FF7A3D"


def page_gradient(p):
    return (
        f"radial-gradient(900px 520px at 92% -6%, rgba(245,197,24,.17), transparent 62%), "
        f"radial-gradient(820px 620px at -6% 104%, rgba(255,122,61,.15), transparent 60%), "
        f"linear-gradient(160deg, {p['page_a']} 0%, {p['page_b']} 100%)"
    )


def css_vars(p):
    return (
        " ".join(f"--{k.replace('_', '-')}: {v};" for k, v in p.items())
        + f" --page: {page_gradient(p)};"
    )


FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,700;1,9..144,400"
    "&family=Inter:wght@400..700&display=swap"
)
DISCLAIMER = (
    "Decision support only. Suggestions come from a model trained on voluntary, unverified ASRS reports "
    "and have not been validated for operational use. A safety analyst must review every result."
)
JS_TOGGLE = (
    "() => { const d = document.body.classList.toggle('dark'); "
    "try { localStorage.setItem('asrs-theme', d ? 'dark' : 'light'); } catch (e) {} }"
)
JS_INIT = (
    "() => { let m = 'dark'; try { m = localStorage.getItem('asrs-theme') || 'dark'; } catch (e) {} "
    "document.body.classList.toggle('dark', m === 'dark'); }"
)

CSS = f"""
@import url('{FONT_URL}');
:root {{ {css_vars(L)} --yellow: {YELLOW}; --orange: {ORANGE}; color-scheme: light; }}
.dark {{ {css_vars(D)} color-scheme: dark; }}
html, body, gradio-app {{ background: var(--page) !important; background-attachment: fixed !important; }}
.gradio-container {{ --layout-gap: 12px; --block-padding: 14px;
                    max-width: 1200px !important; margin: auto; background: transparent !important;
                    font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }}
.html-container {{ padding: 0 !important; }}
.html-container > div:last-child, .html-container > .card:last-child {{ margin-bottom: 0; }}
h1, h2, h3, h4, .serif {{ font-family: 'Fraunces', Georgia, serif !important; color: var(--ink) !important; }}
#mode-toggle {{ position: fixed; top: 14px; right: 18px; z-index: 50; width: auto !important; min-width: 0 !important;
               padding: 6px 14px !important; border-radius: 999px !important; box-shadow: 0 2px 10px rgba(0,0,0,.18); }}
#hero {{ position: relative; overflow: hidden; background: linear-gradient(115deg, #2B2F36 0%, #1B1E23 75%);
         border: 1px solid #3A3F48; border-radius: 18px; padding: 28px 32px 26px 46px; margin: 0 0 4px; }}
#hero::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 14px;
                 background: linear-gradient(180deg, {YELLOW} 0 55%, {ORANGE} 55% 100%); }}
#hero .eyebrow {{ font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; color: {ORANGE};
                  font-weight: 600; margin-bottom: 8px; }}
#hero h1 {{ margin: .2em 0; font-size: 2.6rem; font-weight: 700; line-height: 1.1; color: #F2EFE8 !important; }}
#hero p {{ margin: 0; font-family: 'Fraunces', Georgia, serif; font-style: italic; font-size: 1.15rem; color: #D6D2C8; }}
.pill {{ display: inline-block; margin: 16px 8px 0 0; padding: 4px 12px; border-radius: 999px; font-size: 12.5px;
         font-weight: 600; color: {YELLOW}; background: rgba(245,197,24,.12); border: 1px solid rgba(245,197,24,.45); }}
.card {{ background: var(--panel); border: 1px solid var(--edge); border-radius: 14px; padding: 16px 20px; margin: 0 0 12px;
         color: var(--ink); }}
.card.accent {{ border-left: 5px solid {YELLOW}; }}
.sec {{ font-size: .78rem; font-weight: 600; letter-spacing: .16em; text-transform: uppercase; color: var(--muted);
        margin: 0 0 10px !important; line-height: 1.2; }}
.chip {{ display: inline-block; margin: 2px 6px 2px 0; padding: 3px 11px; border-radius: 999px; font-size: 13px;
         background: var(--chip); color: var(--soft); border: 1px solid var(--edge); }}
.chip.on {{ background: {YELLOW}; color: #1B1E23; border-color: {YELLOW}; font-weight: 700; }}
.chip.off {{ background: transparent; color: var(--muted); }}
.chip.hit {{ background: rgba(245,197,24,.18); color: var(--accent-ink); border-color: rgba(200,150,0,.5); font-weight: 600; }}
.nbr {{ display: flex; gap: 14px; }}
.nbr .rank {{ flex: none; width: 36px; height: 36px; border-radius: 50%; background: {YELLOW}; color: #1B1E23;
              font-family: 'Fraunces', Georgia, serif; font-size: 18px; font-weight: 700;
              display: flex; align-items: center; justify-content: center; }}
.nbr .body {{ flex: 1; min-width: 0; }}
.bar {{ height: 6px; border-radius: 4px; background: var(--track); margin: 6px 0 8px; overflow: hidden; }}
.bar > span {{ display: block; height: 100%; background: linear-gradient(90deg, {YELLOW}, {ORANGE}); }}
.excerpt {{ font-size: 13.5px; line-height: 1.55; color: var(--soft); margin-top: 6px; }}
.card > .excerpt:first-of-type {{ margin-top: 0; }}
.gbar {{ display: grid; grid-template-columns: 150px 1fr 48px; gap: 12px; align-items: center; margin: 10px 0; font-size: 13.5px; }}
.gbar.wide {{ grid-template-columns: 260px 1fr 56px; margin: 8px 0; }}
.gname {{ color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.gtrack {{ position: relative; height: 14px; border-radius: 7px; background: var(--track); }}
.gfill {{ height: 100%; border-radius: 7px; background: var(--off); }}
.gfill.on {{ background: {YELLOW}; }}
.gthr {{ position: absolute; top: -4px; bottom: -4px; width: 3px; margin-left: -1px; border-radius: 2px; background: {ORANGE}; }}
.gval {{ color: var(--soft); text-align: right; font-variant-numeric: tabular-nums; }}
.legend {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
.legend i {{ display: inline-block; width: 3px; height: 11px; background: {ORANGE}; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px; }}
.kpi {{ background: #2B2F36; border: 1px solid #3A3F48; color: #D6D2C8; border-radius: 14px; padding: 16px 20px;
        border-bottom: 4px solid {ORANGE}; }}
.kpi b {{ display: block; font-family: 'Fraunces', Georgia, serif; font-size: 36px; color: {YELLOW}; line-height: 1.1; }}
.kpi span {{ font-size: 12.5px; color: #A6A296; }}
.status {{ display: inline-block; padding: 2px 10px; border-radius: 6px; font-size: 12.5px; font-weight: 600; }}
.status.ok {{ background: #DCFCE7; color: #166534; }}
.status.open {{ background: #FEF3C7; color: #92400E; }}
.status.no {{ background: #FFE6D8; color: #B4400F; }}
.dark .status.ok {{ background: rgba(74,222,128,.15); color: #86efac; }}
.dark .status.open {{ background: rgba(245,197,24,.15); color: {YELLOW}; }}
.dark .status.no {{ background: rgba(255,122,61,.16); color: #FF9A6B; }}
.foot {{ font-family: 'Fraunces', Georgia, serif; font-style: italic; font-size: .92rem; color: var(--muted); padding: 8px 0 0; }}
footer {{ display: none !important; }}
[role="tab"][aria-selected="true"] {{ color: var(--accent-ink) !important; border-color: {YELLOW} !important; }}
.gradio-container .gallery-item, .gradio-container .examples td, .gradio-container .examples button,
.gradio-container table.table tbody td {{ background: var(--panel) !important; color: var(--ink) !important;
                                         border-color: var(--edge) !important; }}
[role="tab"], .label-wrap, .label-wrap span, button.primary, button.secondary, button.lg, button.sm,
.gradio-container label > span, .gradio-container .block-title, .gradio-container span[data-testid="block-info"],
.card b, .nbr b {{ font-family: 'Fraunces', Georgia, serif !important; }}
.excerpt i {{ font-family: 'Fraunces', Georgia, serif; font-size: 1.02em; }}
table.simple {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
table.simple, table.simple tr, table.simple td, table.simple th {{ border: none !important; outline: none !important;
                                                                  box-shadow: none !important; background: transparent !important; }}
table.simple td, table.simple th {{ padding: 8px 10px; border-bottom: 1px solid var(--edge) !important; text-align: left;
                                    color: var(--ink); }}
@media (max-width: 760px) {{ .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} #hero h1 {{ font-size: 32px; }}
                            .gbar {{ grid-template-columns: 110px 1fr 44px; }} .gbar.wide {{ grid-template-columns: 140px 1fr 48px; }} }}
"""

HERO = f"""<div id="hero"><div class="eyebrow">NASA ASRS · MAAI 902 · NLP and LLMs</div><h1>✈ {TITLE}</h1>
<p>Semantic triage and similar-incident retrieval for aviation safety narratives.</p>
<span class="pill">DistilBERT multi-label classifier</span><span class="pill">MiniLM + FAISS retrieval</span>
<span class="pill">BERTopic themes</span><span class="pill">test macro-F1 0.680</span></div>"""

RESULTS = dict(
    f1_bert="0.680",
    f1_tfidf="0.679",
    p5_tfidf="0.678",
    p5_chunk="0.654",
    themes="42",
    cv="0.74",
)


# ---------- compute: classifier ----------
def load_classifier(model_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clf = (
        AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    )
    return clf, device


def prepare_text(raw, expand=False):
    text = " ".join(
        normalise_placeholders(raw).split()
    )  # the training-time preparation
    return expand_abbreviations(text) if expand else text  # expansion = the H3 ablation


def text_diagnostics(text, tokenizer, max_len=512):
    tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return dict(words=len(text.split()), tokens=tokens, truncated=tokens > max_len - 2)


def score_groups(text, clf, tokenizer, device):
    ids = head_tail_ids(text, tokenizer)  # first 128 + last 382 word-pieces
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits = clf(input_ids=x, attention_mask=torch.ones_like(x)).logits[0]
    return torch.sigmoid(logits).cpu().numpy()


def select_groups(probs, thresholds, names, n=3):
    th = np.asarray(thresholds)
    order = np.argsort(probs)[::-1][:n]
    return pd.DataFrame(
        {
            "group": [names[i] for i in order],
            "probability": probs[order],
            "threshold": th[order],
            "predicted": probs[order] >= th[order],
        }
    )


# ---------- compute: retrieval and topics ----------
def retrieve_similar(text, ctx, k=5, chunked=True, group="Any", n_chars=280):
    index, embed = (
        (ctx.index_chunk, embed_chunked)
        if chunked
        else (ctx.index_trunc, embed_truncated)
    )
    fetch = (
        k if group == "Any" else min(len(ctx.df_index), k * 40)
    )  # over-fetch, then filter by group
    scores, ids = search_similar(index, embed(ctx.minilm, [text]), k=fetch)
    scores, ids = scores[0], ids[0]
    if group != "Any":
        keep = ctx.Y_index[ids, ctx.names.index(group)] == 1
        scores, ids = scores[keep][:k], ids[keep][:k]
    rows = ctx.df_index.iloc[ids]
    out = pd.DataFrame(
        {
            "similarity": scores,
            "anomalies": rows["anomaly_raw"].values,
            "excerpt": rows["text"].str[:n_chars].values,
        }
    )
    if "synopsis" in rows.columns:
        out["synopsis"] = rows["synopsis"].values
    return out


def assign_topic(text, ctx, n_words=8, k=15, min_share=0.4):
    """Theme of a new narrative = weighted vote of the themes of its nearest training narratives.
    The 42 themes were fitted once with BERTopic on the training and validation narratives; this lookup reuses
    those labels without the UMAP/HDBSCAN models, so the Space needs no pickled model."""
    emb = embed_truncated(ctx.minilm, [text])
    scores, ids = ctx.index_tv.search(emb, k)
    votes = {}
    for s_, j in zip(scores[0], ids[0]):
        t = int(ctx.topics[j])
        votes[t] = votes.get(t, 0.0) + max(float(s_), 0.0)
    total = sum(votes.values()) or 1.0
    t, w = max(votes.items(), key=lambda kv: kv[1])
    if t == -1 or w / total < min_share:
        return dict(id=-1, words=[], size=None)
    return dict(
        id=t, words=ctx.topic_words[t][:n_words], size=int(ctx.topic_sizes.get(t, 0))
    )


def theme_table(ctx, top=15):
    big = sorted(ctx.topic_sizes.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return pd.DataFrame(
        {
            "topic": [t for t, _ in big],
            "reports": [n for _, n in big],
            "keywords": [", ".join(ctx.topic_words[t][:4]) for t, _ in big],
        }
    )


def theme_examples(topic_id, ctx, n=3):
    return sample_topic_docs(ctx.texts_tv, ctx.topics, topic_id, n=n)


def random_validation_case(ctx):
    i = random.randrange(len(ctx.df_val))
    truth = [ctx.names[j] for j in np.flatnonzero(ctx.Y_val[i] == 1)]
    return ctx.df_val["text"].iloc[i], truth


# ---------- render: HTML pieces ----------
def chip(label, cls=""):
    return f'<span class="chip {cls}">{html.escape(str(label))}</span>'


def render_truth(truth):
    if not truth:
        return ""
    return (
        '<div class="card accent"><div class="sec">Analyst-coded groups (ground truth for this validation report)'
        f"</div>{''.join(chip(g, 'hit') for g in truth)}</div>"
    )


def render_chips(top):
    on = top[top["predicted"]]
    head = (
        ", ".join(on["group"])
        if len(on)
        else "No group clears its threshold; showing the most probable"
    )
    body = "".join(
        chip(f"{g}  {p:.0%}", "on" if ok else "off")
        for g, p, ok in zip(top["group"], top["probability"], top["predicted"])
    )
    return (
        f'<div class="card accent"><div class="sec">Suggested anomaly groups</div>{body}'
        f'<div style="font-size:13px;color:var(--muted);margin-top:8px">{html.escape(head)}</div></div>'
    )


def render_topic(info):
    if info["id"] == -1:
        return (
            '<div class="card"><div class="sec">Recurring theme</div><div class="excerpt" style="font-size:14.5px">'
            "No discovered theme fits this narrative (outlier). About a third of reports fall outside "
            "every theme.</div></div>"
        )
    words = "".join(chip(w, "hit") for w in info["words"])
    return (
        f'<div class="card"><div class="sec">Recurring theme</div><b>Theme {info["id"]}</b> '
        f'<span style="color:var(--muted)">({info["size"]:,} reports in the corpus)</span><div style="margin-top:6px">'
        f"{words}</div></div>"
    )


def render_neighbours(nbrs, predicted=()):
    if nbrs.empty:
        return '<div class="card">No reports matched this filter.</div>'
    cards = []
    for rank, r in enumerate(nbrs.itertuples(), 1):
        groups = [g.strip() for g in str(r.anomalies).split(";") if g.strip()]
        chips = "".join(
            chip(g, "hit" if g.split(":")[0].strip() in predicted else "")
            for g in groups
        )
        syn = getattr(r, "synopsis", "")
        extra = (
            f'<div class="excerpt"><i>{html.escape(str(syn))}</i></div>' if syn else ""
        )
        cards.append(
            f'<div class="card nbr"><div class="rank">{rank}</div><div class="body">'
            f'<div style="display:flex;justify-content:space-between"><b>Similarity {r.similarity:.2f}</b></div>'
            f'<div class="bar"><span style="width:{max(0, min(1, r.similarity)) * 100:.0f}%"></span></div>'
            f'{chips}<div class="excerpt">{html.escape(r.excerpt)}…</div>{extra}</div></div>'
        )
    return '<div class="sec">Most similar past reports</div>' + "".join(cards)


def render_prep(raw, text, diag, expand):
    flag = (
        "truncated: first 128 and last 382 word-pieces are kept"
        if diag["truncated"]
        else "fits in the 512-token window, nothing is cut"
    )
    return (
        f'<div class="card"><div class="sec">Prepared text</div><div class="excerpt">{html.escape(text)}</div>'
        f'<div style="margin-top:10px;color:var(--muted);font-size:13px">{diag["words"]} words, {diag["tokens"]} '
        f"word-pieces; {flag}. Abbreviation expansion is {'on' if expand else 'off'} (the classifier was trained "
        f"on unexpanded text).</div></div>"
    )


def render_results():
    r = RESULTS
    kpis = "".join(
        f'<div class="kpi"><b>{v}</b><span>{k}</span></div>'
        for v, k in [
            (r["f1_bert"], f"DistilBERT test macro-F1 (TF-IDF {r['f1_tfidf']})"),
            (
                r["p5_tfidf"],
                f"precision@5, TF-IDF keywords (MiniLM chunked {r['p5_chunk']})",
            ),
            (r["themes"], "recurring themes found"),
            (r["cv"], "topic coherence (C_v)"),
        ]
    )
    rows = [
        (
            "H1",
            "DistilBERT beats TF-IDF on macro-F1",
            "Not supported: level with the baseline on test",
            "no",
        ),
        (
            "H2",
            "Embedding retrieval beats keyword retrieval",
            "Not supported: keywords better on test",
            "no",
        ),
        ("H3", "Abbreviation expansion helps TF-IDF more", "Not supported", "no"),
    ]
    body = "".join(
        f'<tr><td><b>{a}</b></td><td>{b}</td><td><span class="status {s}">{c}</span></td></tr>'
        for a, b, c, s in rows
    )
    return (
        f'<div class="kpi-row">{kpis}</div><div class="card"><div class="sec">Hypotheses</div>'
        f'<table class="simple">{body}</table></div><div class="foot">Classifier and retrieval: test set, run once. '
        f"Themes and abbreviation expansion: validation data.</div>"
    )


ABOUT = f"""### Intended use
A prototype that helps a safety analyst triage a new narrative, see its recurring theme and find similar past reports.
It supports decisions and never codes reports automatically.

### Limits
- Trained on voluntary, unverified ASRS reports, so it says nothing about how often events happen.
- Rare groups (for example Other) are weak; about a third of reports fit no theme.
- Classifier and retrieval results come from a single test-set run; themes come from training and validation data. Not validated for operational use.

### Safeguards
Analyst in the loop, probabilities shown, no operator-level ranking, no causal or fault claims, no generative text.

*{DISCLAIMER}*"""


# ---------- render: charts (HTML bars, so they follow the light/dark mode) ----------
def render_group_bars(top):
    rows = "".join(
        f'<div class="gbar"><span class="gname">{html.escape(str(g))}</span><div class="gtrack">'
        f'<div class="gfill {"on" if ok else ""}" style="width:{max(0, min(1, p)) * 100:.0f}%"></div>'
        f'<div class="gthr" style="left:{max(0, min(1, t)) * 100:.0f}%" title="tuned threshold {t:.2f}"></div></div>'
        f'<span class="gval">{p:.0%}</span></div>'
        for g, p, t, ok in zip(
            top["group"], top["probability"], top["threshold"], top["predicted"]
        )
    )
    return (
        f'<div class="card"><div class="sec">Probability by group</div>{rows}'
        f'<div class="legend"><i></i>tuned threshold</div></div>'
    )


def render_theme_bars(table):
    top = table["reports"].max()
    rows = "".join(
        f'<div class="gbar wide"><span class="gname">{t} · {html.escape(str(w))}</span><div class="gtrack">'
        f'<div class="gfill on" style="width:{n / top * 100:.0f}%"></div></div><span class="gval">{n}</span></div>'
        for t, w, n in zip(table["topic"], table["keywords"], table["reports"])
    )
    return f'<div class="card"><div class="sec">Reports per theme</div>{rows}</div>'


# ---------- wiring ----------
def make_analyse_fn(ctx):
    def analyse(raw, top_n, thr_mode, expand, retr_mode, k, group):
        if not raw or len(raw.split()) < 10:
            return (
                '<div class="card">Enter a narrative of at least 10 words.</div>',
                "",
                "",
                "",
                "",
            )
        text = prepare_text(raw, expand)
        probs = score_groups(text, ctx.clf, ctx.tokenizer, ctx.device)
        th = (
            ctx.thresholds
            if thr_mode.startswith("Tuned")
            else np.full(len(ctx.names), 0.5)
        )
        top = select_groups(probs, th, ctx.names, int(top_n))
        nbrs = retrieve_similar(
            text, ctx, int(k), retr_mode.startswith("Chunked"), group
        )
        return (
            render_chips(top),
            render_group_bars(top),
            render_topic(assign_topic(text, ctx)),
            render_neighbours(nbrs, set(top.loc[top["predicted"], "group"])),
            render_prep(raw, text, text_diagnostics(text, ctx.tokenizer), expand),
        )

    return analyse


def make_example_fn(ctx):
    def example():
        text, truth = random_validation_case(ctx)
        return text, render_truth(truth)

    return example


def make_theme_fn(ctx, table):
    def show(choice):
        t = int(choice.split("·")[0])
        words = "".join(chip(w, "hit") for w in ctx.topic_words[t][:10])
        docs = "".join(
            f'<div class="card"><div class="excerpt">{html.escape(d)}…</div></div>'
            for d in theme_examples(t, ctx)
        )
        return f'<div class="card accent"><b>Theme {t}</b><div style="margin-top:6px">{words}</div></div>{docs}'

    return show


def make_blocks():
    kw = dict(title=TITLE)
    if (
        "theme" in inspect.signature(gr.Blocks.__init__).parameters
    ):  # gradio < 6 takes theme and css here
        kw.update(theme=THEME, css=CSS)
    return gr.Blocks(**kw)


def build_theme():
    """Light values for the default mode and dark values for the .dark class that the toggle button sets."""

    def values(p):
        return dict(
            body_background_fill=page_gradient(p),
            body_text_color=p["ink"],
            body_text_color_subdued=p["muted"],
            background_fill_primary=p["panel"],
            background_fill_secondary=p["chip"],
            panel_background_fill=p["panel"],
            block_background_fill=p["panel"],
            block_border_color=p["edge"],
            block_label_background_fill=p["panel"],
            block_label_text_color=p["muted"],
            block_title_text_color=p["soft"],
            border_color_primary=p["edge"],
            input_background_fill=p["input"],
            input_border_color=p["edge"],
            input_placeholder_color=p["muted"],
            button_primary_background_fill=YELLOW,
            button_primary_background_fill_hover="#FFD84A",
            button_primary_text_color="#1B1E23",
            button_primary_border_color=YELLOW,
            button_secondary_background_fill=p["panel"],
            button_secondary_background_fill_hover=p["chip"],
            button_secondary_text_color=p["ink"],
            button_secondary_border_color=p["edge"],
            slider_color=YELLOW,
            checkbox_background_color_selected=YELLOW,
            checkbox_border_color_selected=YELLOW,
            link_text_color=p["accent_ink"],
            color_accent_soft="rgba(245,197,24,.15)",
        )

    base = values(L)
    base.update({k + "_dark": v for k, v in values(D).items()})
    return gr.themes.Base(
        primary_hue=gr.themes.colors.yellow,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(**base)


THEME = build_theme()


def build_demo(analyse_fn, example_fn, theme_fn, ctx, table, examples):
    with make_blocks() as demo:
        gr.HTML(HERO)
        toggle = gr.Button("◐  Light / Dark", elem_id="mode-toggle", size="sm")
        toggle.click(fn=None, inputs=None, outputs=None, js=JS_TOGGLE)
        demo.load(fn=None, inputs=None, outputs=None, js=JS_INIT)
        with gr.Tabs():
            with gr.Tab("Triage"):
                with gr.Row():
                    with gr.Column(scale=5):
                        inp = gr.Textbox(
                            lines=11,
                            label="Safety narrative",
                            placeholder="Paste an ASRS narrative (at least 10 words)",
                        )
                        with gr.Row():
                            run = gr.Button("Analyse", variant="primary", min_width=110)
                            rnd = gr.Button("Random report", min_width=130)
                            clr = gr.Button("Clear", min_width=80)
                        truth = gr.HTML()
                        with gr.Accordion("Options", open=True):
                            top_n = gr.Slider(
                                1, 5, value=3, step=1, label="Anomaly groups to show"
                            )
                            k = gr.Slider(
                                3,
                                10,
                                value=5,
                                step=1,
                                label="Similar reports to retrieve",
                            )
                            thr = gr.Radio(
                                ["Tuned per group", "Fixed 0.5"],
                                value="Tuned per group",
                                label="Decision threshold",
                            )
                            retr = gr.Radio(
                                ["Chunked (recommended)", "Truncated"],
                                value="Chunked (recommended)",
                                label="Retrieval input for long narratives",
                            )
                            grp = gr.Dropdown(
                                ["Any"] + ctx.names,
                                value="Any",
                                label="Only show reports with group",
                            )
                            expand = gr.Checkbox(
                                False, label="Expand abbreviations first (ablation)"
                            )
                        gr.Examples(
                            examples=examples, inputs=inp, label="Quick examples"
                        )
                    with gr.Column(scale=6):
                        chips = gr.HTML()
                        plot = gr.HTML()
                        topic = gr.HTML()
                        nbrs = gr.HTML()
                        with gr.Accordion("How this text was prepared", open=False):
                            prep = gr.HTML()
            with gr.Tab("Theme explorer"):
                gr.HTML(
                    '<div class="sec">The 15 largest of the discovered themes</div>'
                )
                gr.HTML(render_theme_bars(table))
                pick = gr.Dropdown(
                    [f"{t} · {w}" for t, w in zip(table["topic"], table["keywords"])],
                    label="Inspect a theme",
                )
                detail = gr.HTML()
                pick.change(theme_fn, pick, detail)
            with gr.Tab("Project results"):
                gr.HTML(render_results())
            with gr.Tab("About and ethics"):
                gr.Markdown(ABOUT)
        gr.HTML(f'<div class="foot">{html.escape(DISCLAIMER)}</div>')

        controls = [inp, top_n, thr, expand, retr, k, grp]
        outs = [chips, plot, topic, nbrs, prep]
        run.click(analyse_fn, controls, outs)
        rnd.click(example_fn, None, [inp, truth]).then(analyse_fn, controls, outs)
        clr.click(lambda: ("", "", "", "", "", "", ""), None, [inp, truth] + outs)
    return demo


def launch_demo(demo, **kw):
    demo.queue(max_size=16)  # a few visitors at once on a Space
    if (
        "css" in inspect.signature(demo.launch).parameters
    ):  # gradio 6 takes theme and css here
        return demo.launch(theme=THEME, css=CSS, **kw)
    return demo.launch(**kw)


# ---------- load everything the demo needs ----------
def load_context(art=ART):
    meta = json.loads((art / "meta.json").read_text())
    clf, device = load_classifier(str(art / "distilbert"))
    tokenizer = AutoTokenizer.from_pretrained(str(art / "distilbert"))
    minilm = SentenceTransformer(
        meta["embedder"], device="cpu"
    )  # downloaded once from the Hub
    df_index = pd.read_parquet(art / "index_reports.parquet")
    df_val = pd.read_parquet(art / "val_reports.parquet")
    return SimpleNamespace(
        clf=clf,
        tokenizer=tokenizer,
        device=device,
        thresholds=np.load(art / "thresholds.npy"),
        names=meta["names"],
        minilm=minilm,
        index_chunk=build_faiss_index(np.load(art / "emb_chunk_index.npy")),
        index_trunc=build_faiss_index(np.load(art / "emb_trunc_index.npy")),
        df_index=df_index,
        Y_index=np.load(art / "Y_index.npy"),
        df_val=df_val,
        Y_val=np.load(art / "Y_val.npy"),
        index_tv=build_faiss_index(np.load(art / "emb_tv.npy")),
        topic_words={int(k): v for k, v in meta["topic_words"].items()},
        topic_sizes={int(k): int(v) for k, v in meta["topic_sizes"].items()},
        texts_tv=pd.read_parquet(art / "texts_tv.parquet")["text"],
        topics=np.load(art / "topics.npy"),
    )


def warm_up(ctx):
    """First calls compile numba code (UMAP) and load weights; do it before the first visitor."""
    text = "Climbing through flight level two four zero, air traffic control issued a descent and TCAS gave an advisory."
    try:
        prepared = prepare_text(text)
        assign_topic(prepared, ctx)
        retrieve_similar(prepared, ctx, k=3)
        score_groups(prepared, ctx.clf, ctx.tokenizer, ctx.device)
    except Exception as exc:  # never block start-up on the warm-up
        print("warm-up skipped:", exc)


def build_app():
    ctx = load_context()
    warm_up(ctx)
    table = theme_table(ctx)
    examples = [[t] for t in ctx.df_val["text"].sample(3, random_state=1).str[:900]]
    return build_demo(
        make_analyse_fn(ctx),
        make_example_fn(ctx),
        make_theme_fn(ctx, table),
        ctx,
        table,
        examples,
    )


demo = build_app()

if __name__ == "__main__":
    launch_demo(demo)
