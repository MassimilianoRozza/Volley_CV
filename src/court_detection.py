import cv2
import numpy as np

class CourtDetector:
    def __init__(self):
        self.manual_points = None

    def set_manual_points(self, points):
        """
        Sets the manually selected court points.
        """
        self.manual_points = points

    def preprocess(self, frame):
        """
        Applies preprocessing steps like grayscale conversion and edge detection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny Edge Detection
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def detect_lines(self, edges):
        """
        Detects lines in the image using Hough Transform.
        """
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)
        return lines

    def draw_lines(self, frame, lines):
        """
        Draws detected lines on the frame.
        """
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def draw_manual_court(self, frame):
        """
        Draws the manually selected court points and lines based on specific topology.
        Structure assumed:
        - Points 0-3: Main Court Perimeter (4 corners)
        - Points 4-5: Attack Line 1 (2 points)
        - Points 6-7: Center Line / Net (2 points)
        - Points 8-9: Attack Line 2 (2 points)
        """
        if not self.manual_points:
            return frame

        # Draw all points
        for point in self.manual_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        # 1. Draw Perimeter (First 4 points)
        if len(self.manual_points) >= 4:
            perimeter_pts = np.array(self.manual_points[:4], np.int32)
            perimeter_pts = perimeter_pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [perimeter_pts], True, (255, 255, 0), 2) # Cyan for perimeter

        # 2. Draw Internal Lines (Pairs after index 3)
        # Iterate from index 4, taking 2 points at a time
        for i in range(4, len(self.manual_points), 2):
            if i + 1 < len(self.manual_points):
                pt1 = self.manual_points[i]
                pt2 = self.manual_points[i+1]
                cv2.line(frame, pt1, pt2, (0, 165, 255), 2) # Orange for internal lines

        return frame

    def process_frame(self, frame):
        """
        Main pipeline for processing a single frame.
        """
        output_frame = frame.copy()
        
        if self.manual_points:
            output_frame = self.draw_manual_court(output_frame)
        else:
            edges = self.preprocess(frame)
            lines = self.detect_lines(edges)
            output_frame = self.draw_lines(output_frame, lines)
            
        return output_frame
