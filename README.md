# Volley_CV: Analisi Computer Vision per la Pallavolo

Questo progetto mira a sviluppare un sistema di analisi video basato sulla Computer Vision per la pallavolo, al fine di estrarre statistiche dettagliate, tracciare giocatori e palla, e riconoscere azioni di gioco.

## Funzionalità Attuali

### Fase 1: Fondamenta e Calibrazione (Completata)
*   **Calibrazione Manuale del Campo:** Interfaccia guidata per selezionare i punti chiave del campo (Perimetro, Linee 3m, Rete).
*   **Persistenza:** Salvataggio e caricamento automatico dei dati di calibrazione per ogni video.
*   **Radar View (Bird's Eye View):** Generazione di una vista tattica dall'alto con proiezione su un campo sintetico (zona libera e area di gioco colorate).
*   **Modificabilità:** Possibilità di rivedere e correggere la calibrazione prima dell'avvio.

## Roadmap Futura

### Fase 2: Tracciamento Core (Giocatori e Palla)
*   **Tracciamento Giocatori (Player Tracking):** Object Detection (YOLO) e Tracking (DeepSORT/ByteTrack).
*   **Tracciamento Palla (Ball Tracking):** Modelli specializzati (TrackNet) e Filtri di Kalman.

### Fase 3: Riconoscimento Azioni
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

Segui le istruzioni a schermo per calibrare il campo. Se esiste già una calibrazione salvata, ti verrà chiesto se vuoi caricarla.