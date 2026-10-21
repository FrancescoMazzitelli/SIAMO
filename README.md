# SIAMO

Sistema di diagnosi medica di bordo basato su **RAG**. L'utente descrive il caso
in italiano; il sistema ne riconosce la **categoria clinica** (traumi, cardiologia,
gastroenterologia, infezioni, neurologia, respiratorio, dermatologia, altro) e
decide da solo se serve la **Glasgow Coma Scale (GCS)** oppure qualche domanda
mirata. La descrizione del caso viene **tradotta in inglese** per interrogare i
manuali di soccorso marittimo (in inglese), mentre tutte le risposte del modello
restano in **italiano**. Containerizzato, architettura a microservizi REST,
un'unica GPU (NVIDIA GTX 1650, 4GB VRAM).

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
│   ├── startup_ingest.py      #   ingestione automatica di data/documents/ all'avvio
│   └── workflow/              #   LangGraph: state, nodes, workflow
├── documents/                 # COMPONENTE: management dei documenti
│   ├── pdf_converter.py       #   PDF -> pagine JPG in streaming (1540px lato lungo)
│   ├── ocr_service.py         #   vision model via Ollama /api/chat -> markdown
│   ├── markdown_splitter.py   #   testo -> paragrafi (titoli # e righe vuote)
│   ├── chunker.py             #   paragrafo -> chunk con overlap
│   └── document_processor.py  #   orchestrazione della pipeline di ingest
├── rag/                       # COMPONENTE: retrieval
│   ├── vector_store.py        #   indicizzazione gerarchica in Qdrant
│   ├── qdrant_utils.py        #   id deterministici + query_points
│   └── hierarchical_retriever.py  # due livelli (riassunti -> chunk), paragrafi + rerank LLM
├── llm/                       # COMPONENTE: integrazione LLM (testo)
├── core/                      # config + modelli di dominio
└── prompts/italy.py           # prompt in italiano (triage, traduzione, RAG, diagnosi, giudice, rerank)

gui/                           # COMPONENTE: GUI Streamlit (porta 8501)
├── app.py                     #   UI in italiano (diagnosi + gestione documenti)
├── Dockerfile
└── requirements.txt

run_ingestion.py               # ingest batch da riga di comando
data/documents/                # manuali di soccorso in inglese, ingeriti in automatico
config/settings.toml           # configurazione esterna (fonte unica dei parametri)
scripts/pull_models.py         # scarica i modelli dichiarati nella configurazione
```

## Pipeline di ingest dei documenti

```
PDF ──┐
      ├─► pdf_converter: pagine -> JPEG in streaming (lato lungo 1540px)
JPG ──┤
      └─► ocr_service: ogni pagina -> maternion/LightOnOCR-2:1b -> Markdown
                        │
                        ▼
      page_furniture: toglie l'intestazione corrente in cima a ogni pagina
      (es. "Chapter 1 FIRST AID 21", "International Medical Guide for Ships") e il
      numero di pagina in fondo, cosi non finiscono incollati ai paragrafi
                        │
                        ▼
      markdown_splitter: un titolo Markdown (##, ###, ...) inizia sempre un nuovo
      paragrafo, quindi una sezione e il suo contenuto restano insieme; i blocchi
      senza titolo vengono accorpati fino a min_paragraph_chars e i paragrafi oltre
      max_paragraph_chars vengono spezzati, cosi il paragrafo genitore recuperato
      dal retrieval entra sempre nel contesto del LLM
                        │
                        ▼
      ogni paragrafo viene indicizzato SU QDRANT APPENA SI CHIUDE (in streaming,
      mentre le pagine successive sono ancora in OCR): embedding del paragrafo e
      dei suoi chunk -> upsert immediato (memoria costante, avanzamento visibile)
                        │
                        ▼
      chunker: paragrafo -> chunk (max 800 char, overlap 200)
                        │
                        ▼
      LLM gemma3:1b: riassunto del documento (in inglese, per il retrieval),
      scritto come punto di livello documento SOLO alla fine (has_document)
```

I due manuali di soccorso marittimo in inglese (`data/documents/`) hanno un text
layer nativo: con `pdf_prefer_text_layer = true` il testo viene estratto
direttamente e l'OCR serve solo per le pagine scansionate. Con il valore di default
(`false`) ogni pagina passa comunque dall'OCR, come richiesto.

Il punto di **livello documento** (il riassunto) viene scritto per ultimo: finché
non c'è, il documento non compare nella ricerca di livello 1, quindi eventuali
punti parziali di un ingest interrotto restano inerti. Ai riavvii successivi il
documento viene rifatto da capo (gli id sono deterministici, quindi l'upsert è
idempotente).

La ricerca RAG lavora in due passaggi: prima confronta la query con i
**riassunti** dei documenti e seleziona i più rilevanti, poi cerca i **chunk**
più pertinenti solo dentro quei documenti, li riconduce ai **paragrafi completi**
genitore e li fa **rerankare al LLM**, tenendo i paragrafi con punteggio più alto.

## Ingest batch da riga di comando

`run_ingestion.py` indicizza un'intera cartella con la stessa pipeline dell'API:

```bash
python run_ingestion.py                 # indicizza data/documents/
python run_ingestion.py --force         # re-indicizza anche i documenti presenti
python run_ingestion.py --reset         # svuota la collezione Qdrant e re-indicizza
python run_ingestion.py --dir docs      # cartella personalizzata
```

## Auto-ingest di data/documents

All'avvio, `diagnosis-api` lancia in background l'ingestione automatica di tutti
i documenti contenuti in `data/documents/` (PDF, immagini o testo): estrazione
pagine, conversione in JPEG, OCR con LightOnOCR-2, paragrafi, chunk e
embeddings in Qdrant. L'ingestione parte appena i modelli Ollama sono caricati e
procede in parallelo alle richieste API; i documenti già indicizzati (doc_id
deterministico dal filename) vengono saltati ai riavvii successivi.

> I due manuali occupano ~716 pagine: la prima ingestione con OCR è lunga (decine
> di minuti). Per ripartire da zero e rimuovere i vecchi documenti sintetici,
> svuota la collezione e re-ingestisci da terminale:
>
> ```bash
> sudo docker compose exec diagnosis-api python run_ingestion.py --reset
> ```

## Pipeline di diagnosi (LangGraph)

```
assess_case (triage: categoria + serve la GCS?) -> translate_case (caso -> inglese)
   -> [se serve la GCS: 3 domande E/V/M, altrimenti 2-3 domande mirate di categoria]
   -> build_rag_query (query in inglese) -> retrieve_documents (2 livelli: riassunti
      -> chunk, poi rerank LLM)
   -> generate_diagnosis (5 ipotesi) -> evaluate_candidates (LLM-as-Judge, 4 criteri)
   -> confidenza < soglia: generate_followup_question -> risposta utente
      -> build_rag_query ... (nuovo ciclo)
   -> confidenza >= soglia / stop utente / max cicli: END
```

Il nodo di **triage** classifica il caso in una categoria clinica e decide se
applicare la Glasgow Coma Scale (sempre obbligatoria per la categoria
neurologica). Se la GCS non serve, vengono poste 2-3 domande mirate specifiche
per la categoria. Il nodo di **traduzione** porta la descrizione del caso e le
risposte in inglese, così la query RAG combacia con i manuali in inglese; tutte
le risposte mostrate all'utente restano in italiano.

Il nodo di generazione produce **cinque ipotesi diagnostiche** differenziate, ognuna
con nome della condizione, confidenza, motivazione basata sui documenti ed esami
consigliati. Per evitare ipotesi ripetute, le diagnosi uguali o troppo simili vengono
scartate e, quando servono altre ipotesi, il sistema le genera fondandole su **documenti
diversi** (uno per ipotesi, escludendo i documenti già usati e le diagnosi già proposte).
Il nodo **LLM-as-Judge** assegna a ogni ipotesi quattro punteggi
indipendenti (pertinenza, supporto documenti, completezza, sicurezza clinica); la
media più alta individua la diagnosi migliore, che diventa la diagnosi principale
mostrata (le cinque restano disponibili nella risposta API).

Se dopo la diagnosi la confidenza è inferiore alla soglia, il sistema formula da solo
una domanda semplice e osservabile, pensata per chi non è medico (per esempio far
sorridere il paziente, alzare entrambe le braccia, parlare, seguire un oggetto con gli
occhi), con alcune risposte rapide tra cui scegliere. Ogni risposta viene integrata nel
contesto (query RAG e prompt di diagnosi) e la diagnosi viene riformulata con una nuova
confidenza. Il ciclo si chiude quando: la confidenza raggiunge la soglia
(`confidence_threshold`, default 0.7), l'utente ferma l'approfondimento, oppure si
raggiunge il numero massimo di cicli (`max_followup_rounds`, default 4).

## Configurazione

Tutti i parametri (modelli, URL dei servizi, chunking, retrieval, soglie, API e
GUI) vivono in **`config/settings.toml`**: e' la fonte unica, modificabile senza
toccare il codice ne' `docker-compose.yml`.

```toml
# modelli
embedding_model = "nomic-embed-text"
llm_model = "gemma3:1b"
ocr_model = "maternion/LightOnOCR-2:1b"

# documenti
documents_dir = "data/documents"
pdf_prefer_text_layer = false
min_paragraph_chars = 250
max_paragraph_chars = 1800

# retrieval
top_k_documents = 5
context_top_k = 5

# diagnosi
candidate_count = 5
confidence_threshold = 0.7
max_followup_rounds = 4

# triage e lingua
auto_gcs_triage = true
translate_input_to_english = true
```

- Il servizio `ollama-setup` legge lo stesso file (`scripts/pull_models.py`) e
  scarica i modelli dichiarati: per cambiare modello basta modificare
  `config/settings.toml`.
- Se cambi `embedding_model` aggiorna anche `embedding_dimension` (768 per
  `nomic-embed-text`) e ricrea la collezione Qdrant.
- Le variabili d'ambiente con lo stesso nome (maiuscolo) hanno **priorita'** sul
  file: utili per override temporanei. Il percorso del file si cambia con
  `SIAMO_CONFIG_FILE`.
- `.env.example` elenca gli override piu' comuni (es. esecuzione in locale su
  `localhost` invece dei nomi dei container).

## Avvio

```bash
sudo docker compose build
sudo docker compose up -d          # primo avvio: download LightOnOCR-2 (~1.5GB)
sudo docker compose logs -f ollama-setup   # monitora il pull dei modelli
```

## GUI (Streamlit)

Dopo l'avvio dello stack, la GUI è su **http://localhost:8501**:

- sezione **Diagnosi**: interfaccia a chat. Nuova sessione → descrivi il problema →
  il sistema mostra la categoria riconosciuta e decide da solo se servono le 3 domande
  GCS oppure 2-3 domande mirate di categoria (pulsanti di risposta) → esecuzione
  diagnosi. Se la confidenza non basta,
  il sistema prosegue con domande semplici e osservabili con risposte rapide
  selezionabili: rispondi finché vuoi, oppure premi **Sono soddisfatto · ferma la diagnosi**
  per chiudere quando preferisci. Ogni ciclo riformula le ipotesi e aggiorna la confidenza;
  il report mostra la categoria, la GCS e la gravità (solo se applicabile), la confidenza,
  le **cinque ipotesi diagnostiche a confronto**
  (la migliore in evidenza con badge **Migliore**, le altre come **Candidato N**, ognuna con
  confidenza, motivazione, esami consigliati e la griglia di valutazione LLM-as-Judge su
  pertinenza, supporto documenti, completezza e sicurezza clinica) e le fonti consultate.
  L'header SIAMO resta ancorato in alto mentre scorre il contenuto. A diagnosi conclusa,
  accanto a **Nuova diagnosi** trovi **Aggiungi informazioni** per integrare referti, terapie
  o altri dettagli nella stessa sessione
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
| `POST /api/v1/diagnosis/{id}/answer` | risposta a domanda GCS o di categoria (1..6) |
| `POST /api/v1/diagnosis/{id}/diagnose` | esegue RAG + diagnosi |
| `POST /api/v1/diagnosis/{id}/followup` | risposta a domanda di approfondimento |
| `POST /api/v1/diagnosis/{id}/stop` | ferma l'approfondimento e consolida |
| `POST /api/v1/diagnosis/ingest/file` | ingest PDF / immagine / testo |
| `POST /api/v1/diagnosis/ingest/texts` | ingest testi raw |

## Test

```bash
python3 -m compileall -q src
```