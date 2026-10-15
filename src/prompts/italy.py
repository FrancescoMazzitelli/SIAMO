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
    "Informazioni aggiuntive dal dialogo:\n{additional_information}\n\n"
    "La query deve essere in italiano, focalizzata su diagnosi differenziale, "
    "protocolli di trattamento e considerazioni cliniche rilevanti. "
    "Restituisci solo la query, senza altro testo."
)

DIAGNOSIS_PROMPT = (
    "Sei un assistente clinico AI specializzato in valutazione neurologica. "
    "Basandoti sulle seguenti informazioni, formula CINQUE ipotesi diagnostiche "
    "differenziali:\n\n"
    "PROBLEMA DESCRITTO: {problem_description}\n"
    "PUNTEGGIO GCS: {gcs_total}/15 (Gravita: {gcs_severity})\n"
    "Dettaglio GCS - Apertura Oculare: {gcs_eye}/4, Risposta Verbale: {gcs_verbal}/5, "
    "Risposta Motoria: {gcs_motor}/6\n\n"
    "INFORMAZIONI AGGIUNTIVE (dal dialogo con il clinico):\n{additional_information}\n\n"
    "INFORMAZIONI RECUPERATE DAL DATABASE:\n{context_documents}\n\n"
    "Integra le informazioni aggiuntive nella valutazione: se presenti, riformula "
    "le ipotesi tenendone conto.\n\n"
    "REGOLA DI DIVERSITA: le cinque ipotesi devono essere CONDIZIONI CLINICHE DIVERSE "
    "tra loro per meccanismo o causa (per esempio vascolare, infettiva, metabolica, "
    "tossica, epilettica). NON ripetere la stessa diagnosi con parole diverse e non "
    "elencare varianti o sinonimi dello stesso disturbo. Ordinale dalla piu probabile "
    "alla meno probabile.\n\n"
    "Per ciascuna ipotesi fornisci:\n"
    "- 'diagnosis': il nome della condizione;\n"
    "- 'confidence': un punteggio di confidenza da 0.0 a 1.0;\n"
    "- 'motivation': la motivazione clinica che cita i documenti recuperati;\n"
    "- 'exams': la lista degli esami consigliati per confermare o escludere la diagnosi.\n\n"
    "Rispondi ESCLUSIVAMENTE in formato JSON valido con la chiave 'candidates': "
    "una lista di ESATTAMENTE 5 oggetti, ognuno con le chiavi "
    "'diagnosis', 'confidence', 'motivation', 'exams'. "
    "La confidence deve essere un float tra 0.0 e 1.0. "
    "Motivazioni ed esami devono essere in italiano."
)

DIAGNOSIS_SINGLE_PROMPT = (
    "Sei un assistente clinico AI specializzato in valutazione neurologica. Devi "
    "formulare UNA sola ipotesi diagnostica che sia DIVERSA da quelle gia proposte.\n\n"
    "PROBLEMA DESCRITTO: {problem_description}\n"
    "PUNTEGGIO GCS: {gcs_total}/15 (Gravita: {gcs_severity})\n"
    "Dettaglio GCS - Apertura Oculare: {gcs_eye}/4, Risposta Verbale: {gcs_verbal}/5, "
    "Risposta Motoria: {gcs_motor}/6\n\n"
    "INFORMAZIONI AGGIUNTIVE (dal dialogo con il clinico):\n{additional_information}\n\n"
    "DOCUMENTO DI RIFERIMENTO PRINCIPALE (fonda qui l'ipotesi):\n{primary_context}\n\n"
    "ALTRI DOCUMENTI DISPONIBILI (solo come contesto, gia coperti da altre ipotesi):\n"
    "{other_context}\n\n"
    "IPOTESI GIA PROPOSTE (da NON ripetere, nemmeno con sinonimi):\n{excluded}\n\n"
    "Proponi una condizione clinica con meccanismo o causa diversa da tutte quelle "
    "gia proposte e coerente con il quadro del paziente, privilegiando quanto emerge "
    "dal documento di riferimento principale.\n\n"
    "Rispondi ESCLUSIVAMENTE in formato JSON valido con le chiavi "
    "'diagnosis', 'confidence', 'motivation', 'exams'. "
    "La confidence deve essere un float tra 0.0 e 1.0. "
    "Motivazione ed esami devono essere in italiano."
)

JUDGE_PROMPT = (
    "Sei un revisore clinico (LLM-as-Judge). Valuta le seguenti ipotesi "
    "diagnostiche rispetto al caso del paziente.\n\n"
    "PROBLEMA DESCRITTO: {problem_description}\n"
    "PUNTEGGIO GCS: {gcs_total}/15 (Gravita: {gcs_severity})\n\n"
    "DOCUMENTI DI SUPPORTO:\n{context_documents}\n\n"
    "IPOTESI DA VALUTARE:\n{candidates}\n\n"
    "Assegna a ogni ipotesi quattro punteggi interi da 1 a 10, indipendenti tra loro:\n"
    "- pertinenza: quanto la diagnosi corrisponde ai sintomi del paziente;\n"
    "- supporto_documenti: quanto e fondata sui testi clinici recuperati;\n"
    "- completezza: se include diagnosi alternative, esami e piano di follow-up;\n"
    "- sicurezza_clinica: assenza di suggerimenti rischiosi o privi di evidenze.\n\n"
    "Rispondi ESCLUSIVAMENTE in formato JSON valido con la chiave 'evaluations': "
    "una lista di oggetti, uno per ogni ipotesi, con le chiavi "
    "'index' (indice dell'ipotesi, a partire da 0), 'pertinenza', 'supporto_documenti', "
    "'completezza', 'sicurezza_clinica' e 'commento' (breve commento migliorativo in italiano)."
)

RERANK_PROMPT = (
    "Sei un revisore di pertinenza clinica. Valuta quanto ciascun paragrafo e "
    "rilevante rispetto alla seguente query diagnostica.\n\n"
    "QUERY: {query}\n\n"
    "PARAGRAFI CANDIDATI:\n{documents}\n\n"
    "Assegna a ogni paragrafo un punteggio di pertinenza da 0 a 10 "
    "(10 = massima pertinenza clinica).\n\n"
    "Rispondi ESCLUSIVAMENTE in formato JSON valido con la chiave 'ranking': "
    "una lista di oggetti con 'index' (indice del paragrafo) e 'score' (0-10)."
)

FOLLOWUP_QUESTION_PROMPT = (
    "Sei un assistente che aiuta una persona NON medica, per esempio un familiare o "
    "un soccorritore, che ha il paziente davanti a se. La diagnosi ha ancora una "
    "confidenza di {confidence:.2f} e serve un dettaglio in piu.\n\n"
    "Diagnosi attuale: {diagnosis}\n"
    "Problema descritto: {problem_description}\n"
    "Punteggio GCS: {gcs_total}/15 (Gravita: {gcs_severity})\n\n"
    "Domande gia poste:\n{asked_questions}\n\n"
    "Formula UNA sola domanda SEMPLICE e concreta, comprensibile a chi non e un "
    "medico. La domanda deve riguardare un segno che si puo osservare o verificare "
    "SUBITO guardando il paziente o chiedendogli un gesto semplice: per esempio "
    "sorridere, alzare entrambe le braccia, stringere le mani, parlare in modo "
    "comprensibile, seguire un oggetto con gli occhi, stare in piedi, rispondere a "
    "una domanda semplice. Non usare termini medici tecnici, non chiedere esami, "
    "referti o anamnesi complessa, non ripetere le domande gia poste.\n\n"
    "Rispondi SOLO in JSON valido con le chiavi: 'question' (il testo della domanda) "
    "e 'options' (da 2 a 4 risposte brevi tra cui scegliere, ad esempio 'Si', 'No', "
    "'Solo un lato', 'Non riesco a valutarlo'). Le opzioni devono essere brevissime."
)

CONFIDENCE_CALIBRATION_PROMPT = (
    "Valuta la qualita dei dati disponibili per la diagnosi:\n"
    "- Problema descritto: {has_description}\n"
    "- Punteggio GCS completo: {has_full_gcs}\n"
    "- Documenti recuperati: {doc_count}\n"
    "Stima una confidenza complessiva tra 0.0 e 1.0 per la diagnosi, "
    "coerente con la qualita dei dati disponibili. "
    "Rispondi solo con il numero float."
)
