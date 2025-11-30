import cv2
import argparse
import sys
import numpy as np
from src.court_detection import CourtDetector
from src.calibration import CalibrationManager
from src.radar import RadarView

def ask_user_choice_cv(question_text, window_name="Confirmation"):
    """
    Displays a question in an OpenCV window and waits for 'y' or 'n' input.
    Returns True for 'y', False for 'n', or if the window is closed.
    """
    # Create a blank image for the question window
    img_height, img_width = 200, 600
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    img.fill(50) # Dark gray background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color_q = (255, 255, 255) # White for question
    font_color_a = (0, 255, 0)     # Green for answer options
    thickness = 2

    # Put question text
    cv2.putText(img, question_text, (20, 50), font, font_scale, font_color_q, thickness)
    cv2.putText(img, "Press 'y' for Yes, 'n' for No", (20, 100), font, font_scale, font_color_a, thickness)

    cv2.imshow(window_name, img)

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('y'):
            cv2.destroyWindow(window_name)
            return True
        elif key == ord('n'):
            cv2.destroyWindow(window_name)
            return False
        
        # Handle window close (X button)
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow(window_name)
                return False # Treat closing as 'No'
        except:
            return False # Window already destroyed or error

def draw_existing_selections(img, selections):
    """
    Draws the parts of the court that are currently defined in 'selections'.
    selections: list of 4 elements. Each element is a list of points or None.
    Index 0: Perimeter (4 pts)
    Index 1: Near 3m (2 pts)
    Index 2: Center/Net (2 pts)
    Index 3: Far 3m (2 pts)
    """
    # Draw Perimeter
    if selections[0] and len(selections[0]) == 4:
        pts = np.array(selections[0], np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 255, 0), 2) # Cyan
        for pt in selections[0]:
            cv2.circle(img, pt, 5, (200, 200, 200), -1)

    # Draw Lines
    for i in range(1, 4):
        if selections[i] and len(selections[i]) == 2:
            pt1 = selections[i][0]
            pt2 = selections[i][1]
            cv2.line(img, pt1, pt2, (0, 165, 255), 2) # Orange
            for pt in selections[i]:
                cv2.circle(img, pt, 5, (200, 200, 200), -1)

def get_points_for_phase(base_frame, num_points, instruction_text, window_name, context_selections):
    """
    Helper function to select a specific number of points for a phase.
    """
    current_points = [] # Points selected in this phase
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_points) < num_points:
                current_points.append((x, y))

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display_frame = base_frame.copy()
        
        # Draw context (other phases)
        draw_existing_selections(display_frame, context_selections)

        # Draw current points
        for pt in current_points:
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            
        # Draw current dynamic connection
        if len(current_points) > 0:
            if num_points == 4 and len(current_points) > 1: # Perimeter preview
                pts = np.array(current_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], False, (0, 255, 255), 1)
            elif num_points == 2 and len(current_points) == 2: # Line preview
                cv2.line(display_frame, current_points[0], current_points[1], (0, 0, 255), 1)

        # Text Overlay
        cv2.putText(display_frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_text = f"Selected: {len(current_points)}/{num_points}. "
        if len(current_points) == num_points:
            status_text += "Press 'c' to CONFIRM."
        else:
            status_text += "Click to select."
        status_text += " 'r': Reset phase. 'q': Quit."
        
        cv2.putText(display_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if len(current_points) == num_points else (200, 200, 200), 2)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('c') and len(current_points) == num_points:
            return current_points # Success
        
        if key == ord('r'):
            current_points = []
            
        if key == ord('q'):
            return None # User cancelled
        
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return None
        except: pass

def select_court_structure(frame):
    """
    Orchestrates the multi-phase court selection process with edit capability.
    """
    window_name = "Volley_CV - Court Definition"
    cv2.namedWindow(window_name)
    
    # Index 0: Perimeter (4 pts)
    # Index 1: Near 3m (2 pts)
    # Index 2: Center/Net (2 pts)
    # Index 3: Far 3m (2 pts)
    selections = [None, None, None, None]
    
    phase_info = [
        {"pts": 4, "msg": "PHASE 1: Select 4 Perimeter Corners"},
        {"pts": 2, "msg": "PHASE 2: Select Near 3m Line (2 pts)"},
        {"pts": 2, "msg": "PHASE 3: Select Center Line / Net (2 pts)"},
        {"pts": 2, "msg": "PHASE 4: Select Far 3m Line (2 pts)"}
    ]

    # Initial Filling Loop
    for i in range(4):
        if selections[i] is None:
            pts = get_points_for_phase(frame, phase_info[i]["pts"], phase_info[i]["msg"], window_name, selections)
            if pts is None: 
                cv2.destroyWindow(window_name)
                return None
            selections[i] = pts

    # Review / Edit Loop
    while True:
        display_frame = frame.copy()
        draw_existing_selections(display_frame, selections)
        
        # Overlay Instructions
        cv2.putText(display_frame, "REVIEW MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'c' to CONFIRM and Start", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "Edit: '1'-Perim, '2'-Near3m, '3'-Net, '4'-Far3m", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('c'):
            break
            
        # Handle Edit Requests
        if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            idx = int(chr(key)) - 1
            print(f"Editing Phase {idx + 1}...")
            
            # Temporarily clear this selection so it's not drawn in context (optional, but cleaner)
            old_selection = selections[idx]
            selections[idx] = None 
            
            new_pts = get_points_for_phase(frame, phase_info[idx]["pts"], f"EDITING: {phase_info[idx]['msg']}", window_name, selections)
            
            if new_pts is not None:
                selections[idx] = new_pts
            else:
                # If user cancelled edit, restore old
                selections[idx] = old_selection

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow(window_name)
                return None
        except: pass

    cv2.destroyWindow(window_name)
    
    # Flatten list
    all_points = []
    for s in selections:
        all_points.extend(s)
        
    return all_points

def main():
    parser = argparse.ArgumentParser(description="Volley_CV: Court Detection")
    parser.add_argument("--input", type=str, help="Path to input video or image")
    args = parser.parse_args()

    if not args.input:
        print("Error: Please provide an input file using --input")
        sys.exit(1)

    detector = CourtDetector()
    radar_view = RadarView()

    # Check if input is image or video
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.input}")
            sys.exit(1)

        # Manual Calibration Step
        manual_points = None
        
        # 1. Try to load existing calibration
        saved_points = CalibrationManager.load_calibration(args.input)
        if saved_points:
            print("Found existing calibration data.")
            use_saved = ask_user_choice_cv("Use saved calibration?", window_name="Load Calibration")
            if use_saved:
                manual_points = saved_points
            else:
                print("Starting manual calibration...")

        # 2. If no points loaded or rejected, run manual selection
        if manual_points is None:
            ret, first_frame = cap.read()
            if ret:
                manual_points = select_court_structure(first_frame)
                
                if manual_points:
                    # Save the newly selected points
                    CalibrationManager.save_calibration(args.input, manual_points)
                else:
                    print("Manual selection cancelled or skipped.")
                
                # Rewind video to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 3. Set points to detector
        if manual_points:
            detector.set_manual_points(manual_points)

        cv2.destroyAllWindows() # Ensure clean state
        window_name = "Volley_CV - Court Detection"
        radar_window = "Volley_CV - Radar View"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = detector.process_frame(frame)
            cv2.putText(processed_frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(window_name, processed_frame)
            
            # Radar / Birds-eye View
            if detector.manual_points:
                birdseye_frame = radar_view.get_warped_frame(frame, detector.manual_points)
                if birdseye_frame is not None:
                    cv2.imshow(radar_window, birdseye_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Handle window close (X button)
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                pass 
        
        cap.release()
    else:
        # Image processing
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"Error: Could not open image {args.input}")
            sys.exit(1)
            
        # Load calibration for image if exists
        saved_points = CalibrationManager.load_calibration(args.input)
        manual_points = None
        
        if saved_points:
             print("Found existing calibration data.")
             use_saved = ask_user_choice_cv("Use saved calibration?", window_name="Load Calibration")
             if use_saved:
                 manual_points = saved_points
        
        if manual_points is None:
             manual_points = select_court_structure(frame)
             if manual_points:
                 CalibrationManager.save_calibration(args.input, manual_points)

        if manual_points:
            detector.set_manual_points(manual_points)

        processed_frame = detector.process_frame(frame)
        cv2.putText(processed_frame, "Press any key to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Volley_CV - Court Detection", processed_frame)
        
        # Also show radar view for image
        if detector.manual_points:
            birdseye_frame = radar_view.get_warped_frame(frame, detector.manual_points)
            if birdseye_frame is not None:
                cv2.imshow("Volley_CV - Radar View", birdseye_frame)
            
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()