"""
YOLOv8 Drone Detection Training Script
======================================
Trains YOLOv8 on custom drone datasets (Roboflow, Kaggle, or local synthetic).
Can be executed locally or on cloud compute nodes.
"""

import os
import argparse
import shutil
import torch
from ultralytics import YOLO
from src.evaluate import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Drone Detection")
    parser.add_argument("--data", type=str, default="configs/drone_data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained model (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="", help="cuda device (0 or 0,1) or cpu")
    parser.add_argument("--project", type=str, default="runs/train", help="Save directory")
    parser.add_argument("--name", type=str, default="drone_yolov8", help="Experiment name")
    return parser.parse_args()


def train():
    args = parse_args()

    # Determine compute device
    if not args.device:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"==================================================")
    print(f"  🚁 YOLOv8 Drone Detection Training Pipeline     ")
    print(f"==================================================")
    print(f"Model:      {args.model}")
    print(f"Data YAML:  {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device:     {device} ({'GPU CUDA' if device != 'cpu' else 'CPU'})")
    print(f"==================================================")

    # Initialize YOLOv8
    model = YOLO(args.model)

    # Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        patience=15,
        save=True,
        plots=True,
        mosaic=1.0,  # Augment small objects
        mixup=0.1
    )

    # Copy best weights to standard weights/ directory
    best_weights_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_weights_path):
        os.makedirs("weights", exist_ok=True)
        dest_weights = "weights/best.pt"
        shutil.copy2(best_weights_path, dest_weights)
        print(f"[SUCCESS] Best weights saved to: {dest_weights}")

        # Run automated evaluation
        print("[INFO] Running evaluation on test split...")
        eval_metrics = evaluate_model(model_path=dest_weights, data_yaml=args.data, split="val", imgsz=args.imgsz)
        print(f"[METRICS] Precision: {eval_metrics.get('precision')} | Recall: {eval_metrics.get('recall')} | mAP50: {eval_metrics.get('mAP_50')}")
    else:
        print("[WARN] Training completed but best.pt could not be located in expected runs path.")


if __name__ == "__main__":
    train()
