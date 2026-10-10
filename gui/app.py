"""Streamlit GUI for the SIAMO system (RAG diagnosis with the Glasgow Coma Scale).

Talks to the diagnosis-api REST endpoints (docker container).
The user-facing text of this app is Italian by design.
"""

import html
import os

import requests
import streamlit as st

API_URL = os.environ.get("SIAMO_API_URL", "http://localhost:8000").rstrip("/")
BASE = f"{API_URL}/api/v1/diagnosis"

st.set_page_config(page_title="SIAMO · Diagnosi GCS RAG", layout="wide")

_CSS = """
<style>
.stApp {
    background: #f4f7fb;
}

/* header */
.app-header {
    background: linear-gradient(135deg, #0e7490 0%, #155e75 100%);
    color: #ffffff;
    padding: 1.1rem 1.6rem;
    border-radius: 14px;
    margin-bottom: 0.4rem;
    box-shadow: 0 4px 14px rgba(14, 116, 144, 0.18);
}
.app-header h1 { color: #ffffff; margin: 0; font-size: 1.7rem; font-weight: 700; }
.app-header p  { color: #cffafe; margin: 0.25rem 0 0; font-size: 0.92rem; }

/* stepper */
.steps { display: flex; gap: 0.5rem; margin: 1rem 0 1.2rem; }
.step {
    flex: 1; text-align: center; padding: 0.42rem 0.5rem;
    border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    color: #64748b; background: #e2e8f0;
}
.step.done   { background: #059669; color: #ffffff; }
.step.active { background: #2563eb; color: #ffffff; }

/* sezioni card */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
.card h3 { margin: 0 0 0.25rem; font-size: 1.02rem; color: #0f172a; }
.card .hint { font-size: 0.82rem; color: #64748b; margin-bottom: 0.75rem; }

/* metriche */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    text-align: center;
}
.metric-card .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; }
.metric-card .value { font-size: 1.45rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem; }
.metric-card .sub   { font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; }

/* barra confidenza */
.conf-track { height: 8px; background: #e2e8f0; border-radius: 999px; margin-top: 0.5rem; overflow: hidden; }
.conf-fill  { height: 100%; background: #2563eb; border-radius: 999px; }

/* badge gravita */
.badge { display: inline-block; padding: 0.18rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.b-grave     { background: #fee2e2; color: #b91c1c; }
.b-moderato  { background: #ffedd5; color: #c2410c; }
.b-lieve     { background: #dcfce7; color: #15803d; }
.b-unknown   { background: #e2e8f0; color: #475569; }

/* box diagnostico */
.diag-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.5rem;
}

/* raccomandazioni */
.rec-list { list-style: none; padding-left: 0; margin: 0.25rem 0 0; }
.rec-list li {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 0.55rem 0.85rem; margin-bottom: 0.45rem; font-size: 0.92rem;
}

/* documenti recuperati */
.doc-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 0.7rem 1rem; margin-bottom: 0.6rem; font-size: 0.88rem;
}
.doc-card .doc-meta { font-size: 0.78rem; color: #64748b; margin-bottom: 0.35rem; }
.doc-card .doc-body { color: #334155; }

/* pulsanti */
.stButton > button { border-radius: 10px; font-weight: 600; }
.stButton > button[kind="primary"] { background: #2563eb; }

/* input */
.stTextArea textarea, .stTextInput input { border-radius: 10px; }

/* radio */
div[role="radiogroup"] label {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 0.45rem 0.9rem; margin-bottom: 0.35rem;
}

/* sidebar */
.side-brand {
    font-size: 1.5rem; font-weight: 800; color: #0e7490; letter-spacing: 0.02em;
}
.side-session {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 0.6rem 0.8rem; font-size: 0.82rem;
}
</style>
"""

_SEV_BADGES = {"grave": "b-grave", "moderato": "b-moderato", "lieve": "b-lieve", "unknown": "b-unknown"}

def _init_state():
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("next_question", None)
    st.session_state.setdefault("question_index", 0)
    st.session_state.setdefault("gcs_complete", False)
    st.session_state.setdefault("description_sent", False)
    st.session_state.setdefault("problem_description", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("last_ingest", None)

def _reset_flow():
    st.session_state["session_id"] = None
    st.session_state["next_question"] = None
    st.session_state["question_index"] = 0
    st.session_state["gcs_complete"] = False
    st.session_state["description_sent"] = False
    st.session_state["result"] = None

def api_post(path: str, json: dict | None = None, files: dict | None = None) -> dict:
    try:
        resp = requests.post(f"{BASE}{path}", json=json, files=files, timeout=900)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("ATTENZIONE: impossibile raggiungere il servizio diagnosis-api. "
                 "E' in esecuzione? (vedi README, sezione Avvio)")
        st.stop()
    except requests.exceptions.HTTPError as exc:
        detail = str(exc)
        try:
            detail = exc.response.json().get("detail", detail)
        except Exception:
            pass
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 404:
            _reset_flow()
            st.warning("Sessione scaduta: avvia una nuova sessione.")
        else:
            st.error(f"Errore API ({status}): {detail}")
        st.stop()

def start_session():
    data = api_post("/start")
    st.session_state["session_id"] = data["session_id"]
    st.session_state["next_question"] = None
    st.session_state["question_index"] = 0
    st.session_state["gcs_complete"] = False
    st.session_state["description_sent"] = False
    st.session_state["result"] = None

def _phase_label() -> str:
    if st.session_state["result"]:
        return "Completata"
    if st.session_state["gcs_complete"]:
        return "Pronta per la diagnosi"
    if st.session_state["next_question"]:
        return "Questionario GCS"
    if st.session_state["description_sent"]:
        return "Descrizione inviata"
    return "Non avviata"

def render_steps():
    steps = [
        ("1", "Descrizione", st.session_state["description_sent"]),
        ("2", "Questionario GCS", st.session_state["gcs_complete"]),
        ("3", "Esito", st.session_state["result"] is not None),
    ]
    done = [s[2] for s in steps]
    active_idx = done.index(False) if False in done else len(steps)
    html_parts = ['<div class="steps">']
    for i, (num, label, is_done) in enumerate(steps):
        cls = "done" if is_done else ("active" if i == active_idx else "")
        html_parts.append(f'<div class="step {cls}">{num} · {label}</div>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

def metric_card(label: str, value: str, badge: str | None = None, sub: str | None = None):
    if badge:
        value_html = f'<span class="badge {_SEV_BADGES.get(badge, "b-unknown")}">{html.escape(value)}</span>'
    else:
        value_html = html.escape(value)
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value_html}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="side-brand">SIAMO</div>', unsafe_allow_html=True)
        st.caption("Diagnosi medica assistita da RAG e Glasgow Coma Scale")
        st.divider()
        st.markdown("**Stato sessione**")
        sid = st.session_state["session_id"]
        fase = _phase_label()
        st.markdown(
            f'<div class="side-session">ID: {html.escape(sid[:10] + "..." if sid else "nessuna")}<br>'
            f"Fase: {html.escape(fase)}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Nuova sessione", key="sb_new", use_container_width=True):
            start_session()
            st.rerun()
        st.divider()
        st.markdown("**Backend**")
        st.caption(f"{API_URL}")
        st.markdown("**Modelli**")
        st.caption("OCR vision + gemma3:1b + nomic-embed-text")
        st.divider()
        st.caption("Stack: ollama, qdrant, fastapi, langgraph")

def render_diagnosi():
    sid = st.session_state["session_id"]
    st.markdown(
        '<div class="app-header"><h1>SIAMO · Sistema di Diagnosi Medica</h1>'
        "<p>Descrivi il caso, rispondi al questionario Glasgow Coma Scale, "
        "ottieni una diagnosi supportata dai documenti indicizzati.</p></div>",
        unsafe_allow_html=True,
    )
    render_steps()
    with st.container(border=True):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**1 · Descrizione del problema**")
        st.markdown(
            '<div class="hint">Sintomi, durata, eta, contesto. Più è dettagliato, '
            "più la ricerca nei documenti sarà mirata.</div>",
            unsafe_allow_html=True,
        )
        desc = st.text_area(
            "Descrivi il problema medico del paziente:",
            value=st.session_state["problem_description"],
            height=90,
            key="description_input",
        )
        if st.button(
            "Invia descrizione", type="primary", use_container_width=True,
            disabled=not desc.strip(),
        ):
            data = api_post(f"/{sid}/description", json={"description": desc.strip()})
            st.session_state["problem_description"] = desc.strip()
            st.session_state["description_sent"] = True
            st.session_state["next_question"] = data
            st.session_state["question_index"] = 1
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**2 · Questionario Glasgow Coma Scale**")
        q = st.session_state["next_question"]
        if q is None and not st.session_state["gcs_complete"]:
            st.markdown(
                '<div class="hint">Invia la descrizione per iniziare le domande GCS.</div>',
                unsafe_allow_html=True,
            )
        elif q is not None and not st.session_state["gcs_complete"]:
            st.markdown(
                f"**Domanda {q['current_step']} di {q['total_steps']}** · {html.escape(q['question'])}",
                unsafe_allow_html=True,
            )
            options = q["options"]
            labels = list(options.values())
            choice = st.radio(
                "Seleziona la risposta:",
                options=labels,
                key=f"gcs_q{st.session_state['question_index']}",
            )
            if st.button("Invia risposta", type="primary", use_container_width=True):
                answer = int([key for key, value in options.items() if value == choice][0])
                data = api_post(f"/{sid}/answer", json={"answer": answer})
                if data.get("is_complete"):
                    st.session_state["gcs_complete"] = True
                    st.session_state["next_question"] = None
                else:
                    st.session_state["next_question"] = data
                    st.session_state["question_index"] += 1
                st.rerun()
        elif st.session_state["gcs_complete"]:
            st.success("Questionario completato: puoi eseguire la diagnosi.")
            if st.button("Esegui diagnosi (RAG + LLM)", type="primary", use_container_width=True):
                data = api_post(f"/{sid}/diagnose")
                st.session_state["result"] = data
                st.rerun()
    result = st.session_state.get("result")
    if result:
        render_esito(result)

def render_esito(result: dict):
    st.markdown("**3 · Esito**")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("GCS totale", f"{result['gcs_total']}/15", sub="punteggio Glasgow")
    with c2:
        metric_card("Gravita", result["gcs_severity"], badge=result["gcs_severity"])
    with c3:
        conf = min(max(float(result["confidence"]), 0.0), 1.0)
        fill = f'<div class="conf-track"><div class="conf-fill" style="width:{conf * 100:.1f}%"></div></div>'
        st.markdown(
            f'<div class="metric-card"><div class="label">Confidenza</div>'
            f'<div class="value">{conf:.0%}</div>{fill}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="diag-box">', unsafe_allow_html=True)
    st.markdown("**Diagnosi**")
    st.markdown(result["diagnosis"])
    st.markdown("</div>", unsafe_allow_html=True)
    recs = result.get("recommendations") or []
    if recs:
        st.markdown("**Raccomandazioni**")
        items = "".join(f"<li>{html.escape(str(r))}</li>" for r in recs)
        st.markdown(f'<ul class="rec-list">{items}</ul>', unsafe_allow_html=True)
    docs = result.get("retrieved_documents") or []
    if docs:
        with st.expander(f"Documenti recuperati ({len(docs)})"):
            for i, doc in enumerate(docs, start=1):
                meta = doc.get("metadata") or {}
                page = meta.get("page", "?")
                score = float(doc.get("score", 0.0))
                header = (
                    f"{i}. Fonte: {html.escape(str(doc.get('source', '?')))} "
                    f" · pagina {html.escape(str(page))} · score {score:.3f}"
                )
                body = html.escape(str(doc.get("content", "")))
                st.markdown(
                    f'<div class="doc-card"><div class="doc-meta">{header}</div>'
                    f'<div class="doc-body">{body}</div></div>',
                    unsafe_allow_html=True,
                )

def render_documenti():
    st.markdown(
        '<div class="app-header"><h1>SIAMO · Gestione Documenti</h1>'
        "<p>Indicizza PDF, immagini o testi: ogni documento viene convertito, "
        "sottoposto a OCR e indicizzato per la ricerca RAG.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Carica un file**")
        up = st.file_uploader(
            "PDF, immagine o file di testo",
            type=["pdf", "jpg", "jpeg", "png", "webp", "txt"],
        )
        if up is not None and st.button("Indicizza file", type="primary", use_container_width=True):
            with st.spinner("Elaborazione in corso (OCR + paragrafi + embeddings)..."):
                files = {"file": (up.name, up.getvalue(), up.type or "application/octet-stream")}
                data = api_post("/ingest/file", files=files)
            st.session_state["last_ingest"] = data
            st.rerun()
    with c2:
        st.markdown("**Oppure incolla del testo**")
        testo = st.text_area(
            "Testo medico da indicizzare: separa i paragrafi con una riga vuota.",
            height=100,
            key="raw_text_input",
        )
        if st.button("Indicizza testo", type="primary", use_container_width=True,
                     disabled=not testo.strip()):
            with st.spinner("Embeddings e indicizzazione in corso..."):
                data = api_post("/ingest/texts", json={"texts": [testo.strip()]})
            st.session_state["last_ingest"] = data
            st.rerun()
    res = st.session_state.get("last_ingest")
    if res is None:
        st.info("Nessun documento indicizzato in questa sessione di lavoro.")
        return
    st.divider()
    st.markdown("**Risultato ingest**")
    if "summary" in res:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Paragrafi", str(res.get("paragraphs", 0)))
        with c2:
            metric_card("Chunk", str(res.get("chunks", 0)))
        with c3:
            metric_card("Punti Qdrant", str(res.get("total_points", 0)))
        st.markdown(
            f"Documento: **{html.escape(str(res.get('filename', '?')))}** · "
            f"id `{html.escape(str(res.get('doc_id', '?')))}`"
        )
        st.markdown('<div class="diag-box">', unsafe_allow_html=True)
        st.markdown("**Riassunto**")
        st.markdown(res.get("summary", ""))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        ingested = res.get("ingested", 0)
        st.success(f"Indicizzati {ingested} testi.")

def main():
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    render_sidebar()
    tab_diag, tab_doc = st.tabs(["Diagnosi", "Documenti"])
    with tab_diag:
        render_diagnosi()
    with tab_doc:
        render_documenti()

if __name__ == "__main__":
    main()
