# Volley_CV

**Volley_CV** is a Computer Vision project designed to analyze volleyball games. It aims to provide detailed statistics, track players and the ball, and recognize game actions using machine learning and computer vision techniques.

## Project Overview

The project is structured into four logical phases:

1.  **Phase 1: Court Detection & Calibration:** Identifying court lines and mapping pixel coordinates to real-world 2D coordinates (Homography).
2.  **Phase 2: Core Tracking:** Tracking players (using YOLO, DeepSORT) and the ball (using TrackNet/Kalman Filters).
3.  **Phase 3: Action Recognition:** Identifying game actions (spikes, blocks, sets) using Pose Estimation (MediaPipe/OpenPose) and temporal analysis.
4.  **Phase 4: Advanced Analytics:** Generating heatmaps, performance metrics, and statistical reports.

## Technical Stack

*   **Language:** Python
*   **Computer Vision:** OpenCV, NumPy
*   **Deep Learning:** PyTorch, TorchVision, Ultralytics (YOLO)
*   **Pose Estimation:** MediaPipe
*   **Tracking:** Norfair, FilterPy (Kalman Filters)
*   **Data Analysis:** Pandas, SciPy, Scikit-learn
*   **Visualization:** Matplotlib, Jupyter Notebooks

## Project Structure

```
Volley_CV/
├── src/
│   ├── __init__.py
│   ├── court_detection.py  # Manual selection and line drawing
│   ├── calibration.py      # Persistence logic (Load/Save JSON)
│   └── radar.py            # Homography and Radar View generation
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
*   (Optional) CUDA-compatible GPU for faster inference in later phases.

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

**Example:**

```bash
python main.py --input Video_test/AVV_U12M__PROMOVOLLEY/MVI_0001.MP4
```

## Development Conventions

*   **Code Style:** Follow standard Python PEP 8 guidelines.
*   **Modularity:** Core logic resides in `src/`, kept separate from the execution script `main.py`.
*   **Environment:** Always use the virtual environment (`venv`) to manage dependencies.

## Current Status (Phase 1 - Completed)

*   **Court Detection:** Moved to a manual calibration approach due to reliability issues with automatic detection on varying video qualities.
*   **Calibration Flow:** Implemented a robust 4-phase manual selection process (Perimeter, Near 3m, Net, Far 3m) with visual feedback and editing capabilities.
*   **Persistence:** Calibration data is saved to JSON files (e.g., `video.mp4.json`) and automatically loaded upon reuse.
*   **Radar View:** A "Bird's Eye View" window shows the rectified court (using Homography) on a synthetic blue/orange background, ready for future player projection.