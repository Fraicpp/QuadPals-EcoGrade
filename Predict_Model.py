from ultralytics import YOLO
import os

# Initialize the model with your trained weights
model = YOLO('objmodel_0.2.pt')

# Configure prediction parameters
predict_params = {
    "source": "C:/Users/turin/Desktop/Opencv/all",  # path to images
    "conf": 0.25,  # confidence threshold
    "iou": 0.45,   # NMS IoU threshold
    "imgsz": 640,  # image size
    "save": True,  # save results
    "save_txt": True,  # save results to *.txt
    "project": "runs/predict",  # save results to project/name
    "name": "exp"  # save results to project/name
}

# Run prediction
results = model.predict(**predict_params)