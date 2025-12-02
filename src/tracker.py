from ultralytics import YOLO
import cv2
import numpy as np
import torch

class TrackWrapper:
    """
    A wrapper class to adapt YOLOv8 native track results to the interface expected by RadarView.
    Mimics the behavior of deep_sort_realtime track objects.
    """
    def __init__(self, track_id, ltrb, conf=None):
        self.track_id = int(track_id)
        self._ltrb = ltrb # [left, top, right, bottom]
        self.conf = conf

    def is_confirmed(self):
        # Native tracking results are generally considered confirmed if they have an ID
        return True

    def to_ltrb(self):
        return self._ltrb

class PlayerTracker:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLOv8 detector with native tracking (ByteTrack).
        
        Args:
            model_path (str): Path to the YOLO model.
        """
        print(f"Initializing YOLOv8 model: {model_path}...")
        
        # Check and print device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device.upper()}")
        
        self.model = YOLO(model_path)
        
        # Volleyball player class ID in COCO dataset is 0 (person)
        self.target_class_id = 0 
        
        # Filter function for ROI (Region of Interest)
        self.roi_filter = None

    def set_roi_filter(self, filter_func):
        """
        Sets a callback function to filter detections based on position.
        """
        self.roi_filter = filter_func

    def detect_and_track(self, frame, conf_threshold=0.3):
        """
        Performs detection and tracking using YOLOv8 native ByteTrack.

        Args:
            frame (np.array): Input video frame.
            conf_threshold (float): Confidence threshold.

        Returns:
            list: List of TrackWrapper objects.
        """
        # Run YOLOv8 tracking
        # persist=True is essential for tracking continuity
        # tracker="bytetrack.yaml" uses the lightweight ByteTrack algorithm
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=[self.target_class_id], 
            conf=conf_threshold, 
            verbose=False,
            imgsz=640
        )
        
        tracks = []
        
        if results and len(results) > 0:
            r = results[0] # We only process the first (and only) frame
            
            if r.boxes and r.boxes.id is not None:
                # Get boxes, confidences, and IDs
                # boxes.xyxy is [x1, y1, x2, y2]
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    track_id = track_ids[i]
                    conf = confs[i]
                    
                    # Apply ROI Filter if set
                    # Calculate feet position (bottom center)
                    feet_x = (x1 + x2) / 2
                    feet_y = y2
                    
                    if self.roi_filter is not None:
                        if not self.roi_filter((feet_x, feet_y)):
                            continue
                            
                    # Create wrapper
                    t = TrackWrapper(track_id, [x1, y1, x2, y2], conf)
                    tracks.append(t)
                    
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
