# presentation_demo_optimized.py

import cv2
from ultralytics import YOLO
import time

MODEL_PATH = 'best.onnx'

CONFIDENCE_THRESHOLD = 0.5

TRASH_CLASS_ID = 0

print(f"Loading optimized ONNX model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'best.onnx' is in the same directory and you have run 'pip install onnxruntime'.")
    exit()

print("Model loaded successfully. Starting camera feed...")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

display_warning = False
warning_end_time = 0

print("\n--- Demo Controls ---")
print("Press 'd' to Detect trash in the current frame.")
print("Press 'q' to Quit the demo.")
print("---------------------\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        key = cv2.waitKey(1) & 0xFF

        # --- TRIGGER DETECTION ---
        if key == ord('d'):
            print("\n'd' key pressed: Running detection...")
            
            results = model(frame, verbose=False)
            
            trash_detected = False
            
            for result in results:
                annotated_frame = result.plot()
                
                for box in result.boxes:
                    # Check if the detected class is 'Trash' and the confidence is high enough
                    if box.cls[0] == TRASH_CLASS_ID and box.conf[0] > CONFIDENCE_THRESHOLD:
                        print(f"  [SUCCESS] Trash detected with confidence: {box.conf[0]:.2f}")
                        trash_detected = True
            
            if trash_detected:
                frame = annotated_frame
                
                display_warning = True
                warning_end_time = time.time() + 4
                
                print("  [ACTION] Simulating audio warning playback.")
                print("  [ACTION] Simulating alert notification to authorities.")
            else:
                print("  [INFO] No trash detected above the confidence threshold.")

        # --- QUIT DEMO ---
        elif key == ord('q'):
            print("'q' key pressed: Exiting the demonstration.")
            break

        
        if display_warning and time.time() < warning_end_time:
            cv2.putText(
                frame, 
                "TRASH DETECTED! ALERTING AUTHORITIES.", 
                (40, 50),                 # Position (X, Y)
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9,                      
                (0, 0, 255),              
                2                         
            )
        else:
            display_warning = False

        cv2.imshow("Garbage Detection Demo (Optimized)", frame)

finally:
    print("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    print("Demo finished.")