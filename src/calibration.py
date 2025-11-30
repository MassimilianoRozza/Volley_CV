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
        Loads calibration points from a JSON file associated with the video.
        Returns a list of tuples (x, y) or None.
        """
        json_path = CalibrationManager.get_json_path(video_path)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Convert lists back to tuples for consistency with OpenCV
                    points = [tuple(pt) for pt in data] 
                    return points
            except Exception as e:
                print(f"Error loading calibration file: {e}")
        return None

    @staticmethod
    def save_calibration(video_path, points):
        """
        Saves calibration points to a JSON file associated with the video.
        """
        json_path = CalibrationManager.get_json_path(video_path)
        try:
            with open(json_path, 'w') as f:
                json.dump(points, f)
            print(f"Calibration saved to {json_path}")
        except Exception as e:
            print(f"Error saving calibration file: {e}")