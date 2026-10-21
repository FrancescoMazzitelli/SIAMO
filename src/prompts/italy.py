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
        "text": "Come risponde motoriamente il paziente? Esegue comandi, localizza il dolore, si ritira, ha flessione anormale, estensione o nessuna risposta?",
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

CASE_CATEGORIES = {
    "trauma": "Traumi e cadute",
    "cardiac": "Problemi cardiaci",
    "gastro": "Problemi gastro-intestinali",
    "infection": "Infezioni",
    "neuro": "Problemi neurologici",
    "respiratory": "Problemi respiratori",
    "skin": "Problemi della pelle",
    "other": "Altro",
}

CASE_CATEGORY_LINES = "\n".join(f"- {key}: {label}" for key, label in CASE_CATEGORIES.items())

CATEGORY_QUESTIONS: dict[str, list[dict]] = {
    "trauma": [
        {
            "id": "trauma_when",
            "category": "Trauma",
            "text": "C'è stata una caduta, un urto o un trauma? Da quanto tempo?",
            "options": {
                1: "Meno di 1 ora",
                2: "Da 1 a 24 ore",
                3: "Più di 24 ore",
                4: "Non lo so",
            },
        },
        {
            "id": "trauma_head",
            "category": "Trauma",
            "text": "Il paziente ha battuto la testa o ha perso conoscenza?",
            "options": {
                1: "No",
                2: "Ha battuto la testa senza perdita di conoscenza",
                3: "Ha perso conoscenza, anche brevemente",
                4: "Non lo so",
            },
        },
        {
            "id": "trauma_lesion",
            "category": "Trauma",
            "text": "C'è una ferita aperta, un sanguinamento o una deformità di un arto?",
            "options": {
                1: "No",
                2: "Ferita o sanguinamento",
                3: "Sospetta frattura o deformità",
                4: "Non lo so",
            },
        },
    ],
    "cardiac": [
        {
            "id": "cardiac_pain",
            "category": "Cardiologico",
            "text": "Il paziente ha dolore al petto? Come lo descrive?",
            "options": {
                1: "Nessun dolore al petto",
                2: "Dolore oppressivo o bruciante",
                3: "Dolore che si irradia a braccio, collo o mandibola",
                4: "Non lo so",
            },
        },
        {
            "id": "cardiac_signs",
            "category": "Cardiologico",
            "text": "Il paziente ha sudorazione, nausea, pallore o difficoltà a respirare?",
            "options": {
                1: "No",
                2: "Uno tra sudorazione, nausea o pallore",
                3: "Più sintomi insieme",
                4: "Non lo so",
            },
        },
        {
            "id": "cardiac_history",
            "category": "Cardiologico",
            "text": "Il paziente ha problemi cardiaci noti o prende farmaci per il cuore?",
            "options": {
                1: "No",
                2: "Sì, ipertensione o cardiopatia",
                3: "Sì, precedente infarto o angina",
                4: "Non lo so",
            },
        },
    ],
    "gastro": [
        {
            "id": "gastro_site",
            "category": "Gastro-intestinale",
            "text": "Dove è il dolore addominale?",
            "options": {
                1: "Bocca dello stomaco",
                2: "Intorno all'ombelico",
                3: "In basso a destra",
                4: "Diffuso o non localizzato",
            },
        },
        {
            "id": "gastro_loss",
            "category": "Gastro-intestinale",
            "text": "Ci sono vomito o diarrea? Da quanto tempo?",
            "options": {
                1: "No",
                2: "Da meno di 24 ore",
                3: "Da più di 24 ore o con sangue",
                4: "Non lo so",
            },
        },
        {
            "id": "gastro_hydration",
            "category": "Gastro-intestinale",
            "text": "Il paziente riesce a bere e a urinare normalmente?",
            "options": {
                1: "Sì",
                2: "Beve poco",
                3: "Non urina da molte ore",
                4: "Non lo so",
            },
        },
    ],
    "infection": [
        {
            "id": "infection_fever",
            "category": "Infezione",
            "text": "Il paziente ha febbre? Quanto alta?",
            "options": {
                1: "Nessuna febbre",
                2: "Febbre fino a 38.5 °C",
                3: "Febbre oltre 38.5 °C",
                4: "Non lo so",
            },
        },
        {
            "id": "infection_local",
            "category": "Infezione",
            "text": "Ci sono segni locali come ferita arrossata, pus o gonfiore?",
            "options": {
                1: "No",
                2: "Rossore o gonfiore",
                3: "Pus o ferita infetta",
                4: "Non lo so",
            },
        },
        {
            "id": "infection_general",
            "category": "Infezione",
            "text": "C'è confusione, rigidità del collo o difficoltà a respirare?",
            "options": {
                1: "No",
                2: "Solo stanchezza o dolori",
                3: "Confusione, rigidità o difficoltà a respirare",
                4: "Non lo so",
            },
        },
    ],
    "respiratory": [
        {
            "id": "resp_effort",
            "category": "Respiratorio",
            "text": "Il paziente ha difficoltà a respirare a riposo?",
            "options": {
                1: "No",
                2: "Solo sotto sforzo",
                3: "Sì, anche a riposo",
                4: "Non lo so",
            },
        },
        {
            "id": "resp_noise",
            "category": "Respiratorio",
            "text": "Il respiro è rumoroso o il paziente fischia mentre respira?",
            "options": {
                1: "No",
                2: "Sibilo o tosse",
                3: "Rumore intenso o respiro molto corto",
                4: "Non lo so",
            },
        },
        {
            "id": "resp_history",
            "category": "Respiratorio",
            "text": "Il paziente ha asma, bronchite o allergie note?",
            "options": {
                1: "No",
                2: "Asma o bronchite",
                3: "Grave allergia o anafilassi in passato",
                4: "Non lo so",
            },
        },
    ],
    "skin": [
        {
            "id": "skin_lesion",
            "category": "Cute",
            "text": "C'è un'eruzione, un rossore o una ferita della pelle?",
            "options": {
                1: "No",
                2: "Rossore o eruzione localizzata",
                3: "Eruzione estesa o vesciche",
                4: "Non lo so",
            },
        },
        {
            "id": "skin_quality",
            "category": "Cute",
            "text": "La cute è calda, dolente o si sta staccando?",
            "options": {
                1: "No",
                2: "Calda e dolente",
                3: "Vesciche o distacco della pelle",
                4: "Non lo so",
            },
        },
        {
            "id": "skin_swelling",
            "category": "Cute",
            "text": "Il paziente ha prurito, gonfiore del viso o delle labbra?",
            "options": {
                1: "No",
                2: "Prurito",
                3: "Gonfiore di viso o labbra",
                4: "Non lo so",
            },
        },
    ],
    "other": [
        {
            "id": "other_onset",
            "category": "Generale",
            "text": "Da quanto tempo è comparso il problema?",
            "options": {
                1: "Meno di 1 ora",
                2: "Da 1 a 24 ore",
                3: "Più di 24 ore",
                4: "Non lo so",
            },
        },
        {
            "id": "other_trend",
            "category": "Generale",
            "text": "Il problema peggiora rapidamente?",
            "options": {
                1: "No, è stabile",
                2: "Peggiora lentamente",
                3: "Peggiora rapidamente",
                4: "Non lo so",
            },
        },
        {
            "id": "other_history",
            "category": "Generale",
            "text": "Il paziente ha altre malattie note o prende farmaci?",
            "options": {
                1: "No",
                2: "Sì, malattie croniche",
                3: "Sì, molti farmaci",
                4: "Non lo so",
            },
        },
    ],
}

PROBLEM_DESCRIPTION_PROMPT = (
    "Sono un assistente medico AI. Per favore, descrivi brevemente il problema "
    "medico del paziente. Includi i sintomi principali, la durata dei sintomi, "
    "l'eta del paziente e qualsiasi informazione rilevante. "
    "Rispondi in modo conciso ma completo."
)

TRIAGE_PROMPT = (
    "Sei un medico di bordo esperto di medicina marittima e di primo soccorso. "
    "Analizza la descrizione del caso e classificalo.\n\n"
    "DESCRIZIONE DEL CASO:\n{problem_description}\n\n"
    "Categorie ammesse:\n{categories}\n\n"
    "Decidi:\n"
    "- 'category': la chiave della categoria piu adatta tra quelle ammesse;\n"
    "- 'requires_gcs': true se il caso richiede la valutazione del livello di "
    "coscienza con la Glasgow Coma Scale (per esempio trauma cranico, alterazione "
    "dello stato di coscienza, sospetto ictus o problema neurologico acuto); false "
    "per tutti gli altri casi (per esempio problemi gastro-intestinali, cutanei o "
    "cardiaci senza perdita di coscienza);\n"
    "- 'reason': una frase breve, in italiano, che spiega la scelta.\n\n"
    "Rispondi ESCLUSIVAMENTE in JSON valido con le chiavi 'category', 'requires_gcs' e 'reason'."
)

TRANSLATE_PROMPT = (
    "Traduci in inglese le informazioni cliniche che seguono, mantenendo tutti i "
    "dettagli (sintomi, tempi, valori, risposte alle domande). Non aggiungere "
    "commenti, diagnosi o interpretazioni. Restituisci solo la traduzione in inglese.\n\n"
    "INFORMAZIONI CLINICHE:\n{case_text}"
)

RAG_QUERY_BUILDER_PROMPT = (
    "Sei un assistente che prepara query di ricerca per una base di conoscenza "
    "medica composta da manuali di primo soccorso in inglese.\n\n"
    "CATEGORIA: {category}\n"
    "CASO CLINICO (in inglese):\n{case_summary}\n\n"
    "Scrivi UNA sola query di ricerca IN INGLESE, concisa, per recuperare i passaggi "
    "piu pertinenti alla diagnosi differenziale, al trattamento immediato e alla "
    "decisione di evacuazione. Restituisci solo la query, senza altro testo."
)

DIAGNOSIS_PROMPT = (
    "Sei un medico di bordo esperto di medicina marittima e di primo soccorso. "
    "Basandoti sulle informazioni del caso e sui documenti recuperati, formula "
    "CINQUE ipotesi diagnostiche differenziali.\n\n"
    "CATEGORIA DEL CASO: {category}\n"
    "CASO CLINICO (in inglese):\n{case_summary}\n\n"
    "DOCUMENTI RECUPERATI (in inglese):\n{context_documents}\n\n"
    "REGOLA DI DIVERSITA: le cinque ipotesi devono essere CONDIZIONI CLINICHE DIVERSE "
    "tra loro per meccanismo o causa (per esempio traumatica, cardiovascolare, "
    "infettiva, metabolica, tossica, neurologica). NON ripetere la stessa diagnosi "
    "con parole diverse e non elencare varianti o sinonimi dello stesso disturbo. "
    "Ordinale dalla piu probabile alla meno probabile.\n\n"
    "Per ciascuna ipotesi fornisci:\n"
    "- 'diagnosis': il nome della condizione;\n"
    "- 'confidence': un punteggio di confidenza da 0.0 a 1.0;\n"
    "- 'motivation': la motivazione clinica che cita i documenti recuperati;\n"
    "- 'exams': la lista degli esami o delle azioni consigliate per confermare o "
    "escludere la diagnosi.\n\n"
    "Rispondi ESCLUSIVAMENTE in formato JSON valido con la chiave 'candidates': "
    "una lista di ESATTAMENTE 5 oggetti, ognuno con le chiavi "
    "'diagnosis', 'confidence', 'motivation', 'exams'. "
    "La confidence deve essere un float tra 0.0 e 1.0. "
    "TUTTO IL TESTO (diagnosi, motivazioni, esami) DEVE ESSERE IN ITALIANO, "
    "anche se il caso e i documenti sono in inglese."
)

DIAGNOSIS_SINGLE_PROMPT = (
    "Sei un medico di bordo esperto di medicina marittima e di primo soccorso. Devi "
    "formulare UNA sola ipotesi diagnostica che sia DIVERSA da quelle gia proposte.\n\n"
    "CATEGORIA DEL CASO: {category}\n"
    "CASO CLINICO (in inglese):\n{case_summary}\n\n"
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
    "TUTTO IL TESTO (diagnosi, motivazione, esami) DEVE ESSERE IN ITALIANO, "
    "anche se il caso e i documenti sono in inglese."
)

JUDGE_PROMPT = (
    "Sei un revisore clinico (LLM-as-Judge). Valuta le seguenti ipotesi "
    "diagnostiche rispetto al caso del paziente.\n\n"
    "CATEGORIA DEL CASO: {category}\n"
    "CASO CLINICO (in inglese):\n{case_summary}\n\n"
    "DOCUMENTI DI SUPPORTO (in inglese):\n{context_documents}\n\n"
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
    "Categoria del caso: {category}\n"
    "Diagnosi attuale: {diagnosis}\n"
    "Caso (in inglese):\n{case_summary}\n\n"
    "Domande gia poste:\n{asked_questions}\n\n"
    "Formula UNA sola domanda SEMPLICE e concreta, comprensibile a chi non e un "
    "medico e coerente con la categoria del caso. La domanda deve riguardare un segno "
    "che si puo osservare o verificare SUBITO guardando il paziente o chiedendogli un "
    "gesto semplice (per esempio respirare, parlare in modo comprensibile, alzare "
    "entrambe le braccia, indicare il punto piu doloroso, bere un sorso). Non usare "
    "termini medici tecnici, non chiedere esami, referti o anamnesi complessa, non "
    "ripetere le domande gia poste.\n\n"
    "Rispondi SOLO in JSON valido con le chiavi: 'question' (il testo della domanda) "
    "e 'options' (da 2 a 4 risposte brevi tra cui scegliere, ad esempio 'Si', 'No', "
    "'Solo un lato', 'Non riesco a valutarlo'). Le opzioni devono essere brevissime."
)

CONFIDENCE_CALIBRATION_PROMPT = (
    "Valuta la qualita dei dati disponibili per la diagnosi:\n"
    "- Descrizione del problema: {has_description}\n"
    "- Risposte strutturate raccolte: {has_answers}\n"
    "- Documenti recuperati: {doc_count}\n"
    "Stima una confidenza complessiva tra 0.0 e 1.0 per la diagnosi, "
    "coerente con la qualita dei dati disponibili. "
    "Rispondi solo con il numero float."
)
