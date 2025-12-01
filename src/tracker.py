from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

class PlayerTracker:
    def __init__(self, model_path='yolov8n.pt', max_age=25):
        """
        Initializes the YOLOv8 detector and DeepSORT tracker.
        
        Args:
            model_path (str): Path to the YOLO model (default: yolov8n.pt for speed).
            max_age (int): Maximum number of frames to keep a track alive without detection.
        """
        print(f"Initializing YOLOv8 model: {model_path}...")
        self.detector = YOLO(model_path)
        
        print("Initializing DeepSORT tracker...")
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            nms_max_overlap=1.0,
            max_iou_distance=0.7,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
        
        # Volleyball player class ID in COCO dataset is 0 (person)
        self.target_class_id = 0 
        
        # Filter function for ROI (Region of Interest)
        # Should accept a tuple (x, y) and return True if valid, False if ignored.
        self.roi_filter = None

    def set_roi_filter(self, filter_func):
        """
        Sets a callback function to filter detections based on position.
        The function should accept (x, y) coordinates and return True to keep, False to discard.
        """
        self.roi_filter = filter_func

    def detect_and_track(self, frame, conf_threshold=0.3):
        """
        Performs detection and tracking on a single frame.

        Args:
            frame (np.array): Input video frame.
            conf_threshold (float): Confidence threshold for YOLO detections.

        Returns:
            list: List of tracks. Each track has properties like .track_id, .to_tlbr().
        """
        # 1. Detection (YOLO)
        results = self.detector(frame, verbose=False, classes=[self.target_class_id], conf=conf_threshold)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xywh[0].cpu().numpy() # x_center, y_center, width, height
                x, y, w, h = b
                
                # Calculate feet position (bottom center)
                feet_x = x
                feet_y = y + (h / 2)
                
                # Apply ROI Filter if set
                if self.roi_filter is not None:
                    if not self.roi_filter((feet_x, feet_y)):
                        continue
                
                # Convert to [left, top, w, h] for DeepSORT
                l = x - w/2
                t = y - h/2
                
                conf = float(box.conf[0].cpu().numpy())
                
                # DeepSORT expects: [[left, top, w, h], confidence, detection_class]
                detections.append([[l, t, w, h], conf, 'player'])

        # 2. Tracking (DeepSORT)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        return tracks

    def draw_tracks(self, frame, tracks):
        """
        Draws bounding boxes and IDs on the frame.
        """
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb() # left, top, right, bottom
            
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID background
            label = f"ID: {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw ID text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
        return frame
