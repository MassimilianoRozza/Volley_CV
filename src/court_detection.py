import cv2
import numpy as np

class CourtDetector:
    def __init__(self):
        pass

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

    def process_frame(self, frame):
        """
        Main pipeline for processing a single frame.
        """
        edges = self.preprocess(frame)
        lines = self.detect_lines(edges)
        output_frame = frame.copy()
        output_frame = self.draw_lines(output_frame, lines)
        return output_frame
