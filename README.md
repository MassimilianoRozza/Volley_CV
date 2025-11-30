# Volley_CV: Analisi Computer Vision per la Pallavolo

Questo progetto mira a sviluppare un sistema di analisi video basato sulla Computer Vision per la pallavolo, al fine di estrarre statistiche dettagliate, tracciare giocatori e palla, e riconoscere azioni di gioco.

## Roadmap delle Funzionalità

Di seguito è presentata una roadmap delle funzionalità principali che verranno implementate, suddivise in fasi logiche.

### Fase 1: Fondamenta e Calibrazione del Campo
Questa fase si concentra sull'abilitazione del sistema a comprendere l'ambiente di gioco.
*   **Court Detection:** Identificazione automatica delle linee del campo di pallavolo in un frame video.
*   **Omografia (Homography):** Calibrazione della prospettiva della telecamera per mappare i pixel video alle coordinate reali del campo 2D (vista dall'alto). Questo è cruciale per la determinazione delle zone e delle posizioni precise.

### Fase 2: Tracciamento Core (Giocatori e Palla)
Questa fase implementa i meccanismi fondamentali per seguire gli elementi chiave del gioco.
*   **Tracciamento Giocatori (Player Tracking):**
    *   Utilizzo di modelli di Object Detection (es. YOLO) per identificare i giocatori.
    *   Implementazione di algoritmi di Tracking (es. DeepSORT, ByteTrack) per assegnare ID univoci e seguire i giocatori tra i frame.
    *   Distinzione tra squadre (es. tramite analisi dei colori delle maglie).
    *   (Futuro) Integrazione di OCR per il riconoscimento dei numeri di maglia e l'associazione ai nomi degli atleti.
*   **Tracciamento Palla (Ball Tracking):**
    *   Implementazione di modelli specifici per il rilevamento di oggetti piccoli e veloci (es. TrackNet).
    *   Utilizzo di tecniche come il Filtro di Kalman per prevedere la traiettoria della palla e gestire le occlusioni.

### Fase 3: Riconoscimento Azioni ed Eventi di Gioco
In questa fase, il sistema inizia a interpretare cosa sta succedendo in campo.
*   **Pose Estimation:** Estrazione dello scheletro dei giocatori (punti chiave come spalle, gomiti, ginocchia) utilizzando modelli come OpenPose o MediaPipe.
*   **Riconoscimento Azioni (Action Recognition):**
    *   Analisi temporale dei dati di pose per identificare azioni di gioco (es. attacco, muro, servizio, palleggio, bagher, difesa).
    *   Classificazione delle azioni tramite reti neurali ricorrenti (es. LSTM) o Video Classification (es. 3D-CNN).

### Fase 4: Analisi Avanzata e Generazione di Statistiche
L'ultima fase si concentra sulla trasformazione dei dati grezzi in insight utili.
*   **Determinazione Zone Campo:** Assegnazione delle posizioni dei giocatori e del punto di contatto della palla alle zone standard del campo (es. Zona 1, 2, 3, 4, 5, 6).
*   **Rilevamento Falli:**
    *   Identificazione di falli specifici (es. fallo di piede sul servizio, invasione a rete) basata su posizioni e movimenti.
*   **Conteggio Punti:** Rilevamento di eventi che portano a un punto (es. palla che tocca terra nel campo avversario, palla fuori campo).
*   **Statistiche Avanzate:**
    *   **Heatmap:** Generazione di mappe di calore per visualizzare le aree più attive dei giocatori.
    *   **Metriche di Performance:** Calcolo di parametri come altezza del salto (per attacchi/muri), velocità della palla, efficienza degli attacchi, ecc.
    *   Generazione di report statistici per singoli atleti e per la squadra.

---

### Stack Tecnologico (Previsione)
*   **Librerie CV/ML:** OpenCV, NumPy, PyTorch/TensorFlow, Ultralytics YOLO (per detection), MediaPipe/OpenPose (per pose estimation).
*   **Tracking:** DeepSORT, ByteTrack, Norfair.
*   **Linguaggio:** Python.

### Sfide Note
*   Gestione delle occlusioni (giocatori che coprono altri giocatori o la palla).
*   Variabilità degli angoli di ripresa e qualità video.
*   Reattività del tracciamento della palla ad alta velocità.
*   Precisione nel riconoscimento delle azioni sottili.
