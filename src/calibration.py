import json
import os

class CalibrationManager:
    def __init__(self):
        pass

    @staticmethod
    def get_json_path(video_path):
        return video_path + ".json"

    @staticmethod
    def load_calibration(video_path):
        """
        Loads calibration points and settings from a JSON file associated with the video.
        Returns a tuple (points, settings).
        points: list of tuples (x, y) or None
        settings: dict or None
        """
        json_path = CalibrationManager.get_json_path(video_path)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                    points = None
                    settings = None
                    
                    if isinstance(data, list):
                        # Legacy format: Just a list of points
                        points = [tuple(pt) for pt in data]
                    elif isinstance(data, dict):
                        # New format: Dict with keys
                        if "points" in data:
                            points = [tuple(pt) for pt in data["points"]]
                        if "settings" in data:
                            settings = data["settings"]
                            
                    return points, settings
            except Exception as e:
                print(f"Error loading calibration file: {e}")
        return None, None

    @staticmethod
    def save_calibration(video_path, points, settings=None):
        """
        Saves calibration points and optional settings to a JSON file.
        """
        json_path = CalibrationManager.get_json_path(video_path)
        try:
            data = {
                "points": points,
                "settings": settings if settings else {}
            }
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Calibration and settings saved to {json_path}")
        except Exception as e:
            print(f"Error saving calibration file: {e}")