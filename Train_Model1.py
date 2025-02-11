from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # Initialize YOLOv8 model
    model = YOLO("yolov8n.pt")
    
    # Training configuration test 1
    model.train(
        data="C:/Users/turin/Desktop/Opencv/QP.PET-2/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        workers=4,
        save=True,
        amp=True,
        lr0=0.01,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        perspective=0.0005,
        box=0.05,
        cls=0.5,
        dfl=1.0,
        patience=20,
        cos_lr=True
    )

if __name__ == '__main__':
    freeze_support()
    main()