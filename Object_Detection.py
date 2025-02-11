import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("V11/Obj_Mv11_0.3.pt")  # Replace with the correct model path

# Open a connection to the webcam (Try 0 if 1 doesn't work)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use cv2.CAP_DSHOW on Windows

# Check if the camera is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set frame dimensions (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height

# Define class names based on your dataset
class_names = ['PET Bottle', 'PET bottle', 'PET color bottle', 'bottle']

# Find the index of "PET"
PET_CLASS_INDEX = class_names.index("PET Bottle")

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.75

# Check Camera if open
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break 

    # Perform object detection
    results = model(frame)

    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            cls = int(box.cls[0])  # Class index
            conf = float(box.conf[0])  # Confidence score

            # Only process if detected class is "PET" AND confidence meets threshold
            if cls == PET_CLASS_INDEX and conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Bounnding box color w/ class name
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Show class name w/ confidence
                label = f"PET {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)

    cv2.imshow("YOLOv11 Object Detection - PET Only (Conf >= 0.85)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()