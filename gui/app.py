"""Streamlit GUI for the SIAMO system (RAG diagnosis with the Glasgow Coma Scale).

Chat-style interface over the diagnosis-api REST endpoints (docker container).
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
    background: linear-gradient(180deg, #f6f9fc 0%, #eef3f9 100%);
}

/* header */
.app-header {
    background: radial-gradient(120% 150% at 0% 0%, #0ea5e9 0%, #0e7490 45%, #0f766e 100%);
    color: #ffffff;
    padding: 1.5rem 1.9rem;
    border-radius: 20px;
    margin-bottom: 0.5rem;
    box-shadow: 0 16px 38px rgba(13, 74, 110, 0.22);
}
.app-header h1 { color: #ffffff; margin: 0; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.01em; }
.app-header p  { color: rgba(255, 255, 255, 0.88); margin: 0.4rem 0 0; font-size: 0.95rem; max-width: 62ch; }

/* stepper */
.steps { display: flex; gap: 0.55rem; margin: 1.1rem 0 1.4rem; }
.step {
    flex: 1; text-align: center; padding: 0.52rem 0.5rem;
    border-radius: 12px; font-size: 0.82rem; font-weight: 600;
    color: #64748b; background: #eef2f7; border: 1px solid #e6edf5;
}
.step.done   { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }
.step.active { background: #0e7490; color: #ffffff; border-color: #0e7490; box-shadow: 0 8px 18px rgba(14, 116, 144, 0.28); }

/* chat */
.msg-row { display: flex; gap: 0.6rem; margin: 0.55rem 0; align-items: flex-start; }
.msg-row.user { justify-content: flex-end; }
.bubble {
    max-width: 78%; padding: 0.72rem 0.98rem; border-radius: 16px;
    font-size: 0.95rem; line-height: 1.55;
}
.bubble.assistant {
    background: #ffffff; border: 1px solid #e6edf5; color: #0f172a;
    border-top-left-radius: 6px; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
}
.bubble.user {
    background: linear-gradient(135deg, #0e7490, #0f766e); color: #ffffff;
    border-top-right-radius: 6px; box-shadow: 0 8px 18px rgba(14, 116, 144, 0.26);
}
.avatar {
    width: 32px; height: 32px; border-radius: 50%; flex: 0 0 32px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 700;
}
.avatar.ai   { background: #e0f2fe; color: #0369a1; }
.avatar.user { background: #0e7490; color: #ffffff; }

/* report */
.report-title {
    font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: #0e7490; font-weight: 700; margin-bottom: 0.65rem;
}

/* metriche */
.metric-card {
    background: #ffffff; border: 1px solid #e6edf5; border-radius: 16px;
    padding: 0.95rem 1.1rem; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    text-align: center; height: 100%;
}
.metric-card .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; font-weight: 700; }
.metric-card .value { font-size: 1.5rem; font-weight: 800; color: #0f172a; margin-top: 0.2rem; }
.metric-card .sub   { font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; }

/* barra confidenza */
.conf-track { height: 8px; background: #eef2f7; border-radius: 999px; margin-top: 0.55rem; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 999px; }

/* badge gravita */
.badge { display: inline-block; padding: 0.2rem 0.72rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }
.b-grave     { background: #fef2f2; color: #b91c1c; }
.b-moderato  { background: #fff7ed; color: #c2410c; }
.b-lieve     { background: #ecfdf5; color: #047857; }
.b-unknown   { background: #eef2f7; color: #475569; }

/* box diagnostico */
.diag-box {
    background: linear-gradient(180deg, #f8fbff, #eef6ff);
    border: 1px solid #dbeafe; border-left: 4px solid #0ea5e9;
    border-radius: 14px; padding: 1rem 1.15rem; margin-top: 0.55rem;
}

/* raccomandazioni */
.rec-list { list-style: none; padding-left: 0; margin: 0.35rem 0 0; }
.rec-list li {
    background: #ffffff; border: 1px solid #e6edf5; border-left: 3px solid #14b8a6;
    border-radius: 12px; padding: 0.6rem 0.9rem; margin-bottom: 0.5rem;
    font-size: 0.92rem; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

/* documenti recuperati */
.doc-card {
    background: #ffffff; border: 1px solid #e6edf5; border-radius: 12px;
    padding: 0.75rem 1rem; margin-bottom: 0.6rem; font-size: 0.88rem;
}
.doc-card .doc-meta { font-size: 0.76rem; color: #64748b; margin-bottom: 0.35rem; }
.doc-card .doc-body { color: #334155; line-height: 1.5; }

/* pulsanti */
.stButton > button {
    border-radius: 12px; font-weight: 600; border: 1px solid #dbe4ee;
    transition: all 0.15s ease;
}
.stButton > button:hover { border-color: #0e7490; color: #0e7490; transform: translateY(-1px); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0e7490, #0f766e); border: none;
    box-shadow: 0 10px 20px rgba(14, 116, 144, 0.26);
}
.stButton > button[kind="primary"]:hover { color: #ffffff; filter: brightness(1.06); }

/* input */
[data-testid="stChatInput"] { border-radius: 14px; }
.stTextArea textarea, .stTextInput input { border-radius: 10px; }

/* sidebar */
.side-brand {
    font-size: 1.7rem; font-weight: 800; letter-spacing: 0.01em;
    background: linear-gradient(135deg, #0e7490, #0f766e);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.side-session {
    background: #ffffff; border: 1px solid #e6edf5; border-radius: 14px;
    padding: 0.7rem 0.9rem; font-size: 0.82rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
}
</style>
"""

_SEV_BADGES = {"grave": "b-grave", "moderato": "b-moderato", "lieve": "b-lieve", "unknown": "b-unknown"}


class SessionExpired(Exception):
    pass


def _init_state():
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("next_question", None)
    st.session_state.setdefault("question_index", 0)
    st.session_state.setdefault("gcs_complete", False)
    st.session_state.setdefault("description_sent", False)
    st.session_state.setdefault("problem_description", "")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_ingest", None)
    st.session_state.setdefault("view", "Diagnosi")


def _reset_flow():
    st.session_state["session_id"] = None
    st.session_state["next_question"] = None
    st.session_state["question_index"] = 0
    st.session_state["gcs_complete"] = False
    st.session_state["description_sent"] = False
    st.session_state["result"] = None
    st.session_state["problem_description"] = ""
    st.session_state["messages"] = []


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
        if exc.response is not None and exc.response.status_code == 404:
            raise SessionExpired()
        status = exc.response.status_code if exc.response is not None else "?"
        st.error(f"Errore API ({status}): {detail}")
        st.stop()


def _add_message(role: str, text: str, kind: str = "text", meta: dict | None = None):
    st.session_state["messages"].append(
        {"role": role, "text": text, "kind": kind, "meta": meta or {}}
    )


def start_session():
    data = api_post("/start", json={})
    st.session_state["session_id"] = data.get("session_id")
    st.session_state["next_question"] = None
    st.session_state["question_index"] = 0
    st.session_state["gcs_complete"] = False
    st.session_state["description_sent"] = False
    st.session_state["result"] = None
    st.session_state["problem_description"] = ""
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "text": "Ciao, sono l'assistente SIAMO. Descrivi il problema del paziente: "
                    "cosa è successo, da quanto tempo e cosa noti adesso.",
            "kind": "text",
            "meta": {},
        }
    ]


def _ensure_session():
    if not st.session_state["session_id"]:
        start_session()


def _flow_state() -> str:
    if not st.session_state["description_sent"]:
        return "description"
    if not st.session_state["gcs_complete"]:
        return "gcs"
    if st.session_state["result"] is None:
        return "diagnose"
    if st.session_state["result"].get("needs_followup"):
        return "followup"
    return "complete"


def _phase_label() -> str:
    return {
        "description": "Descrizione",
        "gcs": "Questionario GCS",
        "diagnose": "Pronta per la diagnosi",
        "followup": "Approfondimento",
        "complete": "Completata",
    }[_flow_state()]


def _post(path: str, json: dict | None = None) -> dict:
    _ensure_session()
    sid = st.session_state["session_id"]
    return api_post(f"/{sid}{path}", json=json)


def _recover_and_notice(action: str):
    _reset_flow()
    start_session()
    _add_message(
        "assistant",
        "La sessione era scaduta ed è stata ripristinata. " + action,
        kind="text",
        meta={},
    )
    st.rerun()


def _handle_description(text: str):
    _ensure_session()
    sid = st.session_state["session_id"]
    try:
        data = api_post(f"/{sid}/description", json={"description": text})
    except SessionExpired:
        _reset_flow()
        start_session()
        sid = st.session_state["session_id"]
        try:
            data = api_post(f"/{sid}/description", json={"description": text})
        except SessionExpired:
            _recover_and_notice("Reinvia la descrizione del caso.")
            return

    _add_message("user", text)
    st.session_state["problem_description"] = text
    st.session_state["description_sent"] = True
    st.session_state["next_question"] = data
    st.session_state["question_index"] = 1
    _add_message(
        "assistant",
        data.get("question", ""),
        kind="gcs_question",
        meta={"options": data.get("options", {})},
    )
    st.rerun()


def _handle_answer(value: int):
    question = st.session_state["next_question"] or {}
    label = (question.get("options") or {}).get(value, str(value))
    try:
        data = _post("/answer", json={"answer": value})
    except SessionExpired:
        _recover_and_notice("Reinvia la descrizione del caso e rispondi di nuovo alle domande.")
        return

    _add_message("user", label)
    if data.get("is_complete"):
        st.session_state["gcs_complete"] = True
        st.session_state["next_question"] = None
        _add_message(
            "assistant",
            "Questionario Glasgow completato. Premi Esegui diagnosi per la valutazione.",
            kind="text",
            meta={},
        )
    else:
        st.session_state["next_question"] = data
        st.session_state["question_index"] += 1
        _add_message(
            "assistant",
            data.get("question", ""),
            kind="gcs_question",
            meta={"options": data.get("options", {})},
        )
    st.rerun()


def _settled_message(data: dict) -> str:
    confidence = float(data.get("confidence", 0.0) or 0.0)
    threshold = float(data.get("confidence_threshold", 0.0) or 0.0)
    if data.get("stopped"):
        return "Perfetto, mi fermo qui: questa è la diagnosi consolidata con le informazioni disponibili."
    if confidence >= threshold:
        return "La confidenza è sufficiente: questa è la diagnosi consolidata."
    return "Ho raggiunto il numero massimo di approfondimenti: questa è la diagnosi consolidata."


def _append_diagnosis(data: dict):
    intro = "Ho aggiornato la diagnosi." if data.get("followup_round", 0) > 0 else "Ecco la prima valutazione."
    _add_message("assistant", intro, kind="diagnosis", meta={"result": data})
    if data.get("needs_followup"):
        question = data.get("followup_question") or "Puoi darmi un altro dettaglio osservando il paziente?"
        _add_message("assistant", question, kind="text", meta={})
    else:
        _add_message("assistant", _settled_message(data), kind="text", meta={})


def _handle_diagnose():
    _add_message("user", "Esegui la diagnosi.", kind="text", meta={})
    try:
        data = _post("/diagnose")
    except SessionExpired:
        _recover_and_notice("Reinvia la descrizione del caso e rispondi di nuovo alle domande.")
        return
    st.session_state["result"] = data
    _append_diagnosis(data)
    st.rerun()


def _handle_followup(text: str):
    _add_message("user", text)
    try:
        data = _post("/followup", json={"answer": text})
    except SessionExpired:
        _recover_and_notice("Reinvia la descrizione del caso e rispondi di nuovo alle domande.")
        return
    st.session_state["result"] = data
    _append_diagnosis(data)
    st.rerun()


def _handle_stop():
    _add_message("user", "Sono soddisfatto, fermati pure.", kind="text", meta={})
    try:
        data = _post("/stop")
    except SessionExpired:
        _recover_and_notice("Reinvia la descrizione del caso e rispondi di nuovo alle domande.")
        return
    st.session_state["result"] = data
    _add_message("assistant", "Va bene, mi fermo qui.", kind="diagnosis", meta={"result": data})
    st.rerun()


def render_steps():
    result = st.session_state["result"]
    steps = [
        ("1", "Descrizione", st.session_state["description_sent"]),
        ("2", "Questionario GCS", st.session_state["gcs_complete"]),
        ("3", "Diagnosi", result is not None),
        ("4", "Approfondimento", bool(result) and not result.get("needs_followup")),
    ]
    done = [s[2] for s in steps]
    active_idx = done.index(False) if False in done else len(steps)
    parts = ['<div class="steps">']
    for i, (num, label, is_done) in enumerate(steps):
        cls = "done" if is_done else ("active" if i == active_idx else "")
        parts.append(f'<div class="step {cls}">{num} · {label}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


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


def _confidence_color(confidence: float, threshold: float) -> str:
    if confidence >= threshold:
        return "#059669"
    if confidence >= max(threshold - 0.2, 0.0):
        return "#d97706"
    return "#dc2626"


def _render_report(result: dict):
    confidence = min(max(float(result.get("confidence", 0.0) or 0.0), 0.0), 1.0)
    threshold = float(result.get("confidence_threshold", 0.0) or 0.0)
    round_no = int(result.get("followup_round", 0) or 0)
    max_rounds = int(result.get("max_followup_rounds", 0) or 0)
    color = _confidence_color(confidence, threshold)

    with st.container(border=True):
        st.markdown('<div class="report-title">Report diagnostico</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("GCS totale", f"{result.get('gcs_total', 0)}/15", sub="punteggio Glasgow")
        with c2:
            severity = result.get("gcs_severity", "unknown")
            metric_card("Gravità", severity, badge=severity)
        with c3:
            fill = (
                f'<div class="conf-track"><div class="conf-fill" '
                f'style="width:{confidence * 100:.1f}%;background:{color}"></div></div>'
            )
            st.markdown(
                f'<div class="metric-card"><div class="label">Confidenza</div>'
                f'<div class="value" style="color:{color}">{confidence:.0%}</div>{fill}</div>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Ciclo di approfondimento {round_no} di {max_rounds} · soglia di confidenza {threshold:.0%}"
        )

        diagnosis = html.escape(str(result.get("diagnosis", ""))).replace("\n", "<br>")
        st.markdown(
            f'<div class="diag-box"><strong>Diagnosi</strong><br>{diagnosis}</div>',
            unsafe_allow_html=True,
        )

        recommendations = result.get("recommendations") or []
        if recommendations:
            st.markdown("**Cosa fare adesso**")
            items = "".join(f"<li>{html.escape(str(r))}</li>" for r in recommendations)
            st.markdown(f'<ul class="rec-list">{items}</ul>', unsafe_allow_html=True)

        documents = result.get("retrieved_documents") or []
        if documents:
            with st.expander(f"Fonti consultate ({len(documents)})"):
                for i, doc in enumerate(documents, start=1):
                    meta = doc.get("metadata") or {}
                    page = meta.get("page", "?")
                    score = float(doc.get("score", 0.0) or 0.0)
                    header = (
                        f"{i}. Fonte: {html.escape(str(doc.get('source', '?')))} "
                        f"· pagina {html.escape(str(page))} · score {score:.3f}"
                    )
                    body = html.escape(str(doc.get("content", "")))
                    st.markdown(
                        f'<div class="doc-card"><div class="doc-meta">{header}</div>'
                        f'<div class="doc-body">{body}</div></div>',
                        unsafe_allow_html=True,
                    )


def _render_message(msg: dict):
    if msg.get("kind") == "diagnosis":
        _render_report(msg.get("meta", {}).get("result", {}))
        return
    role = msg.get("role", "assistant")
    text = html.escape(str(msg.get("text", ""))).replace("\n", "<br>")
    if role == "user":
        st.markdown(
            f'<div class="msg-row user"><div class="bubble user">{text}</div>'
            f'<div class="avatar user">TU</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-row assistant"><div class="avatar ai">AI</div>'
            f'<div class="bubble assistant">{text}</div></div>',
            unsafe_allow_html=True,
        )


def render_chat():
    for msg in st.session_state["messages"]:
        _render_message(msg)


def _render_gcs_options():
    question = st.session_state["next_question"] or {}
    options = question.get("options") or {}
    st.caption(
        f"Scegli la risposta · domanda {question.get('current_step', '?')} "
        f"di {question.get('total_steps', '?')}"
    )
    for key, label in options.items():
        if st.button(
            label,
            key=f"gcs_{question.get('current_step')}_{key}",
            use_container_width=True,
        ):
            _handle_answer(int(key))


def _render_controls():
    state = _flow_state()
    if state == "description":
        prompt = st.chat_input("Descrivi il problema del paziente...", key="chat_description")
        if prompt:
            _handle_description(prompt.strip())
    elif state == "gcs":
        _render_gcs_options()
    elif state == "diagnose":
        if st.button("Esegui diagnosi", type="primary", use_container_width=True, key="btn_diagnose"):
            _handle_diagnose()
    elif state == "followup":
        result = st.session_state["result"] or {}
        options = result.get("followup_options") or []
        round_no = int(result.get("followup_round", 0) or 0)
        if options:
            st.caption("Scegli una risposta")
            cols = st.columns(2)
            for i, option in enumerate(options):
                with cols[i % 2]:
                    if st.button(option, key=f"quick_{round_no}_{i}", use_container_width=True):
                        _handle_followup(option)
        prompt = st.chat_input("Oppure scrivi la risposta...", key="chat_followup")
        if prompt:
            _handle_followup(prompt.strip())
        if st.button("Sono soddisfatto · ferma la diagnosi", key="btn_stop", use_container_width=True):
            _handle_stop()
    else:
        if st.button("Nuova diagnosi", use_container_width=True, key="btn_restart"):
            start_session()
            st.rerun()


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="side-brand">SIAMO</div>', unsafe_allow_html=True)
        st.caption("Diagnosi medica assistita da RAG e Glasgow Coma Scale")
        view = st.radio("Sezione", ["Diagnosi", "Documenti"], key="view")
        st.divider()
        st.markdown("**Stato sessione**")
        sid = st.session_state["session_id"]
        st.markdown(
            f'<div class="side-session">ID: {html.escape(sid[:10] + "..." if sid else "nessuna")}<br>'
            f"Fase: {html.escape(_phase_label())}</div>",
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
    return view


def render_diagnosi():
    _ensure_session()
    st.markdown(
        '<div class="app-header"><h1>SIAMO · Sistema di Diagnosi Medica</h1>'
        "<p>Descrivi il caso e rispondi al questionario Glasgow Coma Scale. Se la confidenza "
        "non è sufficiente, ti faccio qualche domanda semplice da verificare subito sul "
        "paziente, finché la diagnosi è solida oppure decidi tu di fermarti.</p></div>",
        unsafe_allow_html=True,
    )
    render_steps()
    render_chat()
    _render_controls()


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
            "Testo medico da indicizzare: verrà incorporato e reso ricercabile.",
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
    view = render_sidebar()
    if view == "Documenti":
        render_documenti()
    else:
        render_diagnosi()


if __name__ == "__main__":
    main()
