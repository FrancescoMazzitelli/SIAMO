# SIAMO

Sistema di diagnosi medica RAG con questionario **Glasgow Coma Scale (GCS)** come
entry-point per la raccolta informazioni. Containerizzato, architettura a
microservizi REST, un'unica GPU (NVIDIA GTX 1650, 4GB VRAM).

## Architettura

Tutto passa da **un solo container Ollama** (GPU): embeddings, LLM di
generazione testi e il modello **vision OCR**.

| Servizio | Device | Ruolo |
|---|---|---|
| `ollama` | **GPU** | nomic-embed-text (embeddings), gemma3:1b (LLM), maternion/LightOnOCR-2:1b (OCR) |
| `qdrant` | CPU | Vector store (RAG gerarchico) |
| `diagnosis-api` | CPU | FastAPI REST + LangGraph |
| `streamlit-gui` | CPU | GUI web (porta 8501) per interagire col workflow |
| `ollama-setup` | - | One-shot: scarica i modelli all'avvio |

I tre modelli restano **caricati simultaneamente** in Ollama
(`OLLAMA_MAX_LOADED_MODELS=3`, `OLLAMA_KEEP_ALIVE=30m`). Con una VRAM di 4GB
(LightOnOCR-2:1b ≈ 2.6GB con proiettore vision, gemma3:1b ≈ 1GB,
nomic-embed-text ≈ 0.3GB) Ollama fa **offload automatico su CPU** delle layer
che non entrano in GPU (`num_gpu=-1`): le operazioni restano funzionanti, quelle
parzialmente su CPU sono solo più lente.

Il modello `maternion/LightOnOCR-2:1b` (1.5GB, Q8_0) convertisce documenti
(PDF, scansioni, immagini) in testo markdown in ordine di lettura. Il LLM
`gemma3:1b` genera query RAG e diagnosi.

## Struttura del progetto (per componenti)

```
src/
├── main.py                    # entrypoint FastAPI
├── api/                       # controller REST (layer di presentazione)
│   └── endpoints/diagnosis.py
├── application/               # logica dell'app: orchestrazione sessioni + workflow
│   ├── rag_service.py         #   RAGDiagnosisService (sessioni, ingest, diagnosi)
│   └── workflow/              #   LangGraph: state, nodes, workflow
├── documents/                 # COMPONENTE: management dei documenti
│   ├── pdf_converter.py       #   PDF -> pagine JPG (1540px lato lungo)
│   ├── ocr_service.py         #   vision model via Ollama /api/chat -> markdown
│   ├── markdown_splitter.py   #   markdown -> paragrafi (regola riga vuota)
│   ├── chunker.py             #   paragrafo -> chunk con overlap
│   └── document_processor.py  #   orchestrazione della pipeline di ingest
├── rag/                       # COMPONENTE: retrieval
│   ├── vector_store.py        #   indicizzazione gerarchica in Qdrant
│   └── hierarchical_retriever.py  # chunk hit -> paragrafi genitore completi
├── llm/                       # COMPONENTE: integrazione LLM (testo)
├── core/                      # config + modelli di dominio
└── prompts/italy.py           # prompt in italiano (GCS, RAG, diagnosi, OCR)

gui/                           # COMPONENTE: GUI Streamlit (porta 8501)
├── app.py                     #   UI in italiano (diagnosi + gestione documenti)
├── Dockerfile
└── requirements.txt
```

## Pipeline di ingest dei documenti

```
PDF ──┐
      ├─► pdf_converter: pagine -> JPEG (lato lungo 1540px, aspetto invariato)
JPG ──┤
      └─► ocr_service: ogni pagina -> maternion/LightOnOCR-2:1b -> Markdown
                        │
                        ▼
      markdown_splitter: accumula testo finché non incontra un nuovo paragrafo
      (riga vuota) o la fine del documento; ricongiunge i paragrafi tagliati a
      metà frase dal salto pagina; aggancia i titoli (#) al paragrafo seguente
                        │
                        ▼
      chunker: paragrafo -> chunk (max 500 char, overlap 50)
                        │
                        ▼
      LLM gemma3:1b: riassunto del documento
                        │
                        ▼
      Qdrant: 3 livelli indicizzati (documento / paragrafo / chunk)
```

La ricerca RAG colpisce i **chunk** ma restituisce i **paragrafi completi**
genitore (recupero gerarchico).

## Pipeline di diagnosi (LangGraph)

```
collect_description -> ask_gcs (3 domande: E/V/M) -> build_rag_query
   -> retrieve_documents (gerarchico) -> generate_diagnosis -> END
```

## Avvio

```bash
sudo docker compose build
sudo docker compose up -d          # primo avvio: download LightOnOCR-2 (~1.5GB)
sudo docker compose logs -f ollama-setup   # monitora il pull dei modelli
```

## GUI (Streamlit)

Dopo l'avvio dello stack, la GUI è su **http://localhost:8501**:

- tab **Diagnosi**: nuova sessione → descrizione problema → 3 domande GCS
  (selezione click) → esecuzione diagnosi → esito con GCS/gravità/confidenza,
  raccomandazioni e documenti recuperati
- tab **Documenti**: upload di PDF/immagine/testo (ingest completo OCR +
  paragrafi + embeddings) oppure incolla testo e vedi il riassunto

Setup (solo a build/ricreazione): `sudo docker compose up -d --build streamlit-gui`
(il volume `./gui:/app` monta il codice live, quindi le modifiche a `gui/app.py`
si riflettono con un refresh della pagina).

## API principali

| Endpoint | Descrizione |
|---|---|
| `POST /api/v1/diagnosis/start` | avvia sessione |
| `POST /api/v1/diagnosis/{id}/description` | descrizione problema |
| `POST /api/v1/diagnosis/{id}/answer` | risposta domanda GCS (1..6) |
| `POST /api/v1/diagnosis/{id}/diagnose` | esegue RAG + diagnosi |
| `POST /api/v1/diagnosis/ingest/file` | ingest PDF / immagine / testo |
| `POST /api/v1/diagnosis/ingest/texts` | ingest testi raw |

## Test

```bash
python3 -m compileall -q src
python3 - <<'EOF'
import src.tests.test_markdown_splitter as t
# esegue gli unit test dello splitter markdown->paragrafi
EOF
```