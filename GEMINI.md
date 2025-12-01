# Volley_CV

**Volley_CV** is a Computer Vision project designed to analyze volleyball games. It aims to provide detailed statistics, track players and the ball, and recognize game actions using machine learning and computer vision techniques.

## TODO
- [ ] Add advanced analytics and heatmaps (Phase 4).
- [ ] Refine ball tracking accuracy.

## Project Overview

The project is structured into four logical phases:

1.  **Phase 1: Court Detection & Calibration:** Identifying court lines and mapping pixel coordinates to real-world 2D coordinates (Homography).
2.  **Phase 2: Core Tracking:** Tracking players (using YOLO, DeepSORT) and the ball (using TrackNet/Kalman Filters). **(In Progress)**
3.  **Phase 3: Action Recognition:** Identifying game actions (spikes, blocks, sets) using Pose Estimation (MediaPipe/OpenPose) and temporal analysis.
4.  **Phase 4: Advanced Analytics:** Generating heatmaps, performance metrics, and statistical reports.

## Technical Stack

*   **Language:** Python
*   **Computer Vision:** OpenCV, NumPy
*   **Deep Learning:** PyTorch, TorchVision, Ultralytics (YOLO)
*   **Pose Estimation:** MediaPipe
*   **Tracking:** DeepSORT, FilterPy (Kalman Filters)
*   **Data Analysis:** Pandas, SciPy, Scikit-learn
*   **Visualization:** Matplotlib, Jupyter Notebooks

## Project Structure

```
Volley_CV/
├── src/
│   ├── __init__.py
│   ├── court_detection.py  # Manual selection and line drawing
│   ├── calibration.py      # Persistence logic (Load/Save JSON)
│   ├── radar.py            # Homography, Radar View, and Interactive UI
│   ├── tracker.py          # Player tracking (YOLO + DeepSORT) & ROI filtering
├── venv/                   # Python Virtual Environment
├── .gitignore
├── GEMINI.md               # This context file
├── main.py                 # Entry point and coordination
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Getting Started

### Prerequisites

*   Python 3.x installed.
*   (Optional) CUDA-compatible GPU for faster inference.

### Installation

1.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```

2.  **Activate the Environment:**
    *   Linux/macOS: `source venv/bin/activate`
    *   Windows: `.\venv\Scripts\activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

The project is executed via `main.py`. You need to provide an input video or image.

**Basic Usage:**

```bash
python main.py --input <path_to_video_or_image>
```

## Development Conventions

*   **Code Style:** Follow standard Python PEP 8 guidelines.
*   **Modularity:** Core logic resides in `src/`, kept separate from the execution script `main.py`.
*   **Environment:** Always use the virtual environment (`venv`) to manage dependencies.

## Current Status

*   **Court Detection:** Manual calibration with visual "Radar Guide" to assist in point selection order.
*   **Calibration Flow:** Robust 4-phase selection (Perimeter, Near 3m, Net, Far 3m) with persistence (save/load JSON).
*   **Radar View:**
    *   Top-down "Bird's Eye View" of the court.
    *   **Interactive Controls:**
        *   `SWAP SIDES`: Rotates player positions 180 degrees (useful if camera is on the opposite side).
        *   `MIRROR LR`: Flips player positions horizontally.
*   **Player Tracking (Phase 2):**
    *   **YOLOv8 + DeepSORT:** Integrated for real-time player detection and tracking.
    *   **ROI Filtering:** Automatically excludes detections outside the active playing area (e.g., spectators) based on the calibration.
    *   **Orientation Support:** Supports "Vertical" (baseline view) and "Horizontal" (sideline view) video inputs.
