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
│   ├── startup_ingest.py      #   ingestione automatica di synthetic_data/ all'avvio
│   └── workflow/              #   LangGraph: state, nodes, workflow
├── documents/                 # COMPONENTE: management dei documenti
│   ├── pdf_converter.py       #   PDF -> pagine JPG (1540px lato lungo)
│   ├── ocr_service.py         #   vision model via Ollama /api/chat -> markdown
│   ├── markdown_splitter.py   #   markdown -> paragrafi (regola: titoli #)
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

synthetic_data/                # casi clinici sintetici, ingeriti in automatico
```

## Pipeline di ingest dei documenti

```
PDF ──┐
      ├─► pdf_converter: pagine -> JPEG (lato lungo 1540px, aspetto invariato)
JPG ──┤
      └─► ocr_service: ogni pagina -> maternion/LightOnOCR-2:1b -> Markdown
                        │
                        ▼
      markdown_splitter: accumula il testo finché non incontra un titolo di
      qualsiasi livello (#, ##, ###, ####, #####, ######); il titolo chiude il
      paragrafo corrente e ne apre uno nuovo (titolo incluso). Un paragrafo può
      proseguire anche oltre il salto pagina
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

## Auto-ingest di synthetic_data

All'avvio, `diagnosis-api` lancia in background l'ingestione automatica di tutti
i documenti contenuti in `synthetic_data/` (PDF, immagini o testo): estrazione
pagine, conversione in JPEG, OCR con LightOnOCR-2, paragrafi, chunk e
embeddings in Qdrant. L'ingestione parte appena i modelli Ollama sono caricati e
procede in parallelo alle richieste API; i documenti già indicizzati (doc_id
deterministico dal filename) vengono saltati ai riavvii successivi.

## Pipeline di diagnosi (LangGraph)

```
collect_description -> ask_gcs (3 domande: E/V/M) -> build_rag_query
   -> retrieve_documents (gerarchico) -> generate_diagnosis
   -> confidenza < soglia: generate_followup_question -> risposta utente
      -> build_rag_query ... (nuovo ciclo)
   -> confidenza >= soglia / stop utente / max cicli: END
```

Se dopo la diagnosi la confidenza è inferiore alla soglia, il sistema formula da solo
una domanda semplice e osservabile, pensata per chi non è medico (per esempio far
sorridere il paziente, alzare entrambe le braccia, parlare, seguire un oggetto con gli
occhi), con alcune risposte rapide tra cui scegliere. Ogni risposta viene integrata nel
contesto (query RAG e prompt di diagnosi) e la diagnosi viene riformulata con una nuova
confidenza. Il ciclo si chiude quando: la confidenza raggiunge la soglia
(`CONFIDENCE_THRESHOLD`, default 0.7), l'utente ferma l'approfondimento, oppure si
raggiunge il numero massimo di cicli (`MAX_FOLLOWUP_ROUNDS`, default 4).

## Avvio

```bash
sudo docker compose build
sudo docker compose up -d          # primo avvio: download LightOnOCR-2 (~1.5GB)
sudo docker compose logs -f ollama-setup   # monitora il pull dei modelli
```

## GUI (Streamlit)

Dopo l'avvio dello stack, la GUI è su **http://localhost:8501**:

- sezione **Diagnosi**: interfaccia a chat. Nuova sessione → descrivi il problema → 3
  domande GCS (pulsanti di risposta) → esecuzione diagnosi. Se la confidenza non basta,
  il sistema prosegue con domande semplici e osservabili con risposte rapide
  selezionabili: rispondi finché vuoi, oppure premi **Sono soddisfatto · ferma la diagnosi**
  per chiudere quando preferisci. Ogni ciclo riformula la diagnosi e aggiorna la confidenza;
  il report mostra GCS, gravità, confidenza, cosa fare adesso e fonti consultate
- sezione **Documenti**: upload di PDF/immagine/testo (ingest completo OCR +
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
| `POST /api/v1/diagnosis/{id}/followup` | risposta a domanda di approfondimento |
| `POST /api/v1/diagnosis/{id}/stop` | ferma l'approfondimento e consolida |
| `POST /api/v1/diagnosis/ingest/file` | ingest PDF / immagine / testo |
| `POST /api/v1/diagnosis/ingest/texts` | ingest testi raw |

## Test

```bash
python3 -m compileall -q src
```