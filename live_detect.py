import cv2
from ultralytics import YOLO
import time
import math
import os
import numpy as np

# --- Configuration ---
# We will now use only your single, custom model trained on litter.
TRASH_MODEL_PATH = 'best.onnx' # Your custom model for trash detection in ONNX format

# Confidence threshold for a detection to be considered valid.
CONFIDENCE_THRESHOLD = 0.65

# The Class ID for 'Trash' in your single-class model is likely 0.
TRASH_CLASS_ID = 0

# --- OPTIMIZATION: FRAME SKIPPING ---
FRAME_SKIP = 1

# --- Tracking Configuration ---
COOLDOWN_PERIOD = 5 # Seconds to wait before another alert can be triggered

# --- NEW: Thermal Anomaly Simulation ---
# An alert will only trigger if the trash's color intensity is significantly
# different from its immediate surroundings.
THERMAL_DIFFERENCE_THRESHOLD = 25 # Min difference in pixel intensity (0-255) to be considered an anomaly.

# --- NEW: Floor Detection Configuration ---
# Define a Y-coordinate for the "floor level".
# An alert will ONLY trigger for trash with its bottom edge below this line.
# Adjust this value based on your camera's perspective.
FLOOR_Y_LEVEL = 200


# --- Setup ---
print(f"Loading trash detection model from: {TRASH_MODEL_PATH}")
if not os.path.exists(TRASH_MODEL_PATH):
    print(f"[ERROR] Trash model not found at '{TRASH_MODEL_PATH}'. Please ensure it's in the directory.")
    exit()
trash_model = YOLO(TRASH_MODEL_PATH)

print("Model loaded successfully. Starting live camera feed...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- State Management Variables ---
frame_count = 0
status_text = "Monitoring..."
text_color = (0, 255, 0)
last_alert_time = 0
in_cooldown = False

print("\n--- Live Littering Event Detection is Running (Single Model with Floor Detection & Thermal Check) ---")
print("Press 'q' in the video window to Quit.")
print("-----------------------------------------------------------------------------------------------\n")

def check_thermal_anomaly(frame, box, margin=15, threshold=THERMAL_DIFFERENCE_THRESHOLD):
    """
    Simulates a thermal check by comparing the average color intensity of the object
    with its immediate surroundings.
    Returns True if the difference is significant (an "anomaly").
    """
    try:
        # Convert frame to grayscale for simpler intensity analysis
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = map(int, box)

        # Define the object's region of interest (ROI)
        object_roi = gray_frame[y1:y2, x1:x2]
        if object_roi.size == 0: return False
        object_mean_intensity = np.mean(object_roi)

        # Define the surrounding region with a margin
        h, w = gray_frame.shape
        sx1 = max(0, x1 - margin)
        sy1 = max(0, y1 - margin)
        sx2 = min(w, x2 + margin)
        sy2 = min(h, y2 + margin)
        surrounding_roi = gray_frame[sy1:sy2, sx1:sx2]

        # Create a mask to exclude the object's area from the surrounding calculation
        mask = np.ones_like(surrounding_roi, dtype=bool)
        mask[(y1 - sy1):(y2 - sy1), (x1 - sx1):(x2 - sx1)] = False
        
        # Calculate the mean intensity of only the surrounding pixels
        surrounding_pixels = surrounding_roi[mask]
        if surrounding_pixels.size == 0: return False
        surrounding_mean_intensity = np.mean(surrounding_pixels)

        difference = abs(object_mean_intensity - surrounding_mean_intensity)
        return difference > threshold
    except Exception as e:
        print(f"[WARN] Thermal check failed: {e}")
        return False


# --- Main Demo Loop ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break

        frame_count += 1
        annotated_frame = frame.copy()

        # --- Draw the Floor Line for Visualization ---
        cv2.line(annotated_frame, (0, FLOOR_Y_LEVEL), (640, FLOOR_Y_LEVEL), (255, 255, 0), 2)
        cv2.putText(annotated_frame, "FLOOR LEVEL", (10, FLOOR_Y_LEVEL - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        if in_cooldown:
            if (time.time() - last_alert_time) > COOLDOWN_PERIOD:
                in_cooldown = False
                print("[INFO] Cooldown finished. Resuming monitoring.")
            else:
                status_text = "EVENT DETECTED! (Cooldown)"
                text_color = (0, 0, 255)

        if not in_cooldown and frame_count % FRAME_SKIP == 0:
            # FIX: Changed 'confidence' to 'conf' as per Ultralytics API
            trash_results = trash_model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD) 
            plotted_frame = trash_results[0].plot(img=annotated_frame)
            
            # The results object contains detected boxes
            boxes = trash_results[0].boxes.xyxy.cpu()
            clss = trash_results[0].boxes.cls.cpu().tolist()

            for box, cls in zip(boxes, clss):
                if int(cls) == TRASH_CLASS_ID:
                    # Check if the bottom of the trash object is below the floor line
                    bottom_y = int(box[3])
                    is_on_floor = bottom_y > FLOOR_Y_LEVEL

                    if is_on_floor:
                        # Final check: simulate thermal anomaly
                        has_thermal_anomaly = check_thermal_anomaly(frame, box)
                        
                        if has_thermal_anomaly:
                            in_cooldown = True
                            last_alert_time = time.time()
                            print(f"  [EVENT] Littering Confirmed on Floor! (Thermal Anomaly Detected)")
                            
                            # --- ALERT WITH BEEP SOUND ---
                            print('\a') # This prints the ASCII bell character, producing a system beep.
                            
                            # Break the loop after the first confirmed event to avoid multiple alerts in one frame
                            break
            
            annotated_frame = plotted_frame

            status_text = "Monitoring..."
            text_color = (0, 255, 0)
        
        cv2.putText(annotated_frame, status_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        cv2.imshow("Live Littering Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    print("Demo finished.")

