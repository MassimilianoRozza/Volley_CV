import cv2
import argparse
import sys
from src.court_detection import CourtDetector

def main():
    parser = argparse.ArgumentParser(description="Volley_CV: Court Detection")
    parser.add_argument("--input", type=str, help="Path to input video or image")
    args = parser.parse_args()

    if not args.input:
        print("Error: Please provide an input file using --input")
        sys.exit(1)

    detector = CourtDetector()

    # Check if input is image or video
    # Simple check based on extension for now
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.input}")
            sys.exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = detector.process_frame(frame)

            cv2.imshow("Volley_CV - Court Detection", processed_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    else:
        # Image processing
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"Error: Could not open image {args.input}")
            sys.exit(1)
            
        processed_frame = detector.process_frame(frame)
        cv2.imshow("Volley_CV - Court Detection", processed_frame)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
