# Volley_CV: Analisi Computer Vision per la Pallavolo

Questo progetto mira a sviluppare un sistema di analisi video basato sulla Computer Vision per la pallavolo, al fine di estrarre statistiche dettagliate, tracciare giocatori e palla, e riconoscere azioni di gioco.

## Funzionalità Attuali

### Fase 1: Fondamenta e Calibrazione (Completata)
*   **Calibrazione Manuale del Campo:** Interfaccia guidata per selezionare i punti chiave del campo (Perimetro, Linee 3m, Rete).
*   **Persistenza:** Salvataggio e caricamento automatico dei dati di calibrazione per ogni video.
*   **Radar View (Bird's Eye View):** Generazione di una vista tattica dall'alto con proiezione su un campo sintetico (zona libera e area di gioco colorate).
*   **Modificabilità:** Possibilità di rivedere e correggere la calibrazione prima dell'avvio.

### Fase 2: Tracciamento Core (In Corso)
*   **Tracciamento Giocatori (Player Tracking):** Utilizzo di YOLOv8 e DeepSORT per identificare e tracciare i giocatori nel tempo.
*   **Filtraggio ROI (Region of Interest):** Possibilità di selezionare la "Zona Attiva" (Sinistra, Destra o Entrambi) per tracciare solo i giocatori in campo ed escludere panchine o spettatori.
*   **Supporto Orientamento Video:** Supporta video ripresi sia da fondo campo ("Verticale") che lateralmente ("Orizzontale").
*   **Radar View Interattiva:**
    *   Generazione di una vista tattica 2D dall'alto.
    *   **Correzione Live:** Pulsanti a schermo **"SWAP SIDES"** (Ruota 180°) e **"MIRROR LR"** (Specchia) per correggere la proiezione se i giocatori appaiono nella metà campo errata.

## Roadmap Futura

### Fase 3: Riconoscimento Azioni
*   **Tracciamento Palla (Ball Tracking):** Modelli specializzati (TrackNet) e Filtri di Kalman.
*   **Pose Estimation:** MediaPipe/OpenPose per lo scheletro dei giocatori.
*   **Action Recognition:** Classificazione delle azioni (schiacciata, muro, bagher).

### Fase 4: Statistiche
*   **Heatmap:** Mappe di calore delle posizioni.
*   **Report:** Statistiche avanzate.

---

## Utilizzo

```bash
source venv/bin/activate
python main.py --input percorso/al/video.mp4
```

1.  **Setup**: Segui le istruzioni a terminale per selezionare l'orientamento del video e la zona attiva di tracciamento.
2.  **Calibrazione**: Se non presente, esegui la calibrazione cliccando i punti richiesti sul video.
3.  **Analisi**: Durante la riproduzione, usa i pulsanti nella finestra "Radar" per correggere l'orientamento della mappa se necessario.
