from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # Initialize YOLOv8 model
    model = YOLO("yolov8n.pt")
    
        # Training configuration test 3
    model.train(
        data="C:/Users/turin/Desktop/Opencv/QP.PET-2/data.yaml",  
        epochs=100,  # Reduced to prevent overfitting
        imgsz=320,  # Smaller image size for efficiency (adjust based on embedded device constraints)
        batch=16,  # Adjust based on GPU memory
        workers=4,  # Optimize data loading speed
        save=True,  
        amp=True,  # Automatic Mixed Precision for faster training
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate (reduced decay for small datasets)
        momentum=0.937,  # Optimal SGD momentum
        weight_decay=0.0005,  # Regularization to prevent overfitting
        warmup_epochs=3.0,  # Warmup period for stable training
        warmup_momentum=0.8,  # Gradual increase in momentum
        warmup_bias_lr=0.1,  # Adjust bias learning rate during warmup
        hsv_h=0.015,  # Hue augmentation (±1.5%)
        hsv_s=0.7,  # Saturation augmentation (±70%)
        hsv_v=0.4,  # Value augmentation (±40%)
        degrees=5.0,  # Random rotation (±5 degrees)
        translate=0.1,  # Translation (±10%)
        scale=0.5,  # Scale variation (50-150%)
        shear=2.0,  # Shear distortion (±2 degrees)
        flipud=0.1,  # Vertical flip probability (10%)
        fliplr=0.5,  # Horizontal flip probability (50%)
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.1,  # Mixup augmentation (reduced probability)
        perspective=0.0005,  # Slight perspective transformation
        box=0.05,  # Box loss gain
        cls=0.5,  # Classification loss gain
        dfl=1.0,  # Distribution focal loss gain
        patience=10,  # Early stopping if no improvement
        cos_lr=True,  # Cosine learning rate decay for smoother convergence
        pretrained=True,  # Use pretrained weights for better initialization
        optimizer="auto",  # Automatically select the best optimizer
        label_smoothing=0.1,  # Add label smoothing to improve generalization
        
    )
if __name__ == '__main__':
    freeze_support()
    main()