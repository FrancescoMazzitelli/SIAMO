GCS_QUESTIONS = [
    {
        "id": "eye",
        "category": "Apertura Oculare",
        "text": "Come risponde il paziente con gli occhi? L'apertura oculare è spontanea, a voce, al dolore o assente?",
        "options": {
            1: "Nessuna apertura oculare (1)",
            2: "Apertura a stimolo doloroso (2)",
            3: "Apertura alla voce (3)",
            4: "Apertura spontanea (4)",
        },
    },
    {
        "id": "verbal",
        "category": "Risposta Verbale",
        "text": "Come risponde verbalmente il paziente? La risposta è articolata, confusa, inappropriata, incomprensibile o assente?",
        "options": {
            1: "Nessuna risposta verbale (1)",
            2: "Suoni incomprensibili (2)",
            3: "Parole inappropriate (3)",
            4: "Risposta confusa (4)",
            5: "Risposta orientata (5)",
        },
    },
    {
        "id": "motor",
        "category": "Risposta Motoria",
        "text": "Come risponde motoriamente il paziente? Esegue comandi, localizza il dolore, si ritrae, ha flessione anormale, estensione o nessuna risposta?",
        "options": {
            1: "Nessuna risposta motoria (1)",
            2: "Estensione anormale (2)",
            3: "Flessione anormale (3)",
            4: "Ritirata dal dolore (4)",
            5: "Localizzazione del dolore (5)",
            6: "Obbedisce ai comandi (6)",
        },
    },
]

PROBLEM_DESCRIPTION_PROMPT = (
    "Sono un assistente medico AI. Per favore, descrivi brevemente il problema "
    "medico del paziente. Includi i sintomi principali, la durata dei sintomi, "
    "l'eta del paziente e qualsiasi informazione rilevante. "
    "Rispondi in modo conciso ma completo."
)

GCS_FOLLOWUP_PROMPT = (
    "Dato il problema descritto: '{problem_description}', "
    "la categoria di valutazione attuale e '{category}', "
    "il punteggio GCS parziale e {current_score}. "
    "Fornisci una breve spiegazione clinica di cosa indica questo punteggio "
    "nel contesto del problema descritto dall'utente. Rispondi in italiano, "
    "in modo conciso e chiaro, max 2-3 frasi."
)

RAG_QUERY_BUILDER_PROMPT = (
    "Dato il seguente problema clinico e i risultati della valutazione GCS, "
    "costruisci una query di ricerca ottimale per recuperare informazioni "
    "rilevanti dal database medico.\n\n"
    "Problema descritto: {problem_description}\n"
    "Punteggio GCS: {gcs_total}/15 (Gravita: {gcs_severity})\n"
    "Dettaglio: E={gcs_eye}, V={gcs_verbal}, M={gcs_motor}\n\n"
    "La query deve essere in italiano, focalizzata su diagnosi differenziale, "
    "protocolli di trattamento e considerazioni cliniche rilevanti. "
    "Restituisci solo la query, senza altro testo."
)

DIAGNOSIS_PROMPT = (
    "Sei un assistente clinico AI specializzato in valutazione neurologica. "
    "Basandoti sulle seguenti informazioni, formula una valutazione diagnostica:\n\n"
    "PROBLEMA DESCRITTO: {problem_description}\n"
    "PUNTEGGIO GCS: {gcs_total}/15 (Gravita: {gcs_severity})\n"
    "Dettaglio GCS - Apertura Oculare: {gcs_eye}/4, Risposta Verbale: {gcs_verbal}/5, "
    "Risposta Motoria: {gcs_motor}/6\n\n"
    "INFORMAZIONI RECUPERATE DAL DATABASE:\n{context_documents}\n\n"
    "Fornisci:\n"
    "1. DIAGNOSI: Una valutazione diagnostica differenziale basata sui dati clinici.\n"
    "2. CONFIANZA: Un punteggio di confidenza da 0.0 a 1.0 sulla precisione della "
    "diagnosi, considerando la completezza dei dati disponibili.\n"
    "3. RACCOMANDAZIONI: Una lista di raccomandazioni cliniche prioritarie.\n\n"
    "Rispondi ESCLUSIVAMENTE in formato JSON valido con le chiavi: "
    "'diagnosis', 'confidence', 'recommendations'. "
    "La confidence deve essere un float tra 0.0 e 1.0. "
    "Le raccomandazioni devono essere una lista di stringhe in italiano."
)

CONFIDENCE_CALIBRATION_PROMPT = (
    "Valuta la qualita dei dati disponibili per la diagnosi:\n"
    "- Problema descritto: {has_description}\n"
    "- Punteggio GCS completo: {has_full_gcs}\n"
    "- Documenti recuperati: {doc_count}\n"
    "Suggerisci un fattore di confidenza tra 0.0 e 1.0. "
    "Rispondi solo con il numero float."
)
