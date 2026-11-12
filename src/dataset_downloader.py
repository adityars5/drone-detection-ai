"""
Dataset Downloader & Generator Module
=====================================
Downloads drone detection datasets from Roboflow Universe or Kaggle,
and generates local synthetic datasets with YOLOv8 annotations for testing.
"""

import os
import shutil
import random
from typing import Optional
import cv2
import numpy as np
from src.simulated_feed import SimulatedDroneCamera


ROBOFLOW_POPULAR_DATASETS = [
    {
        "name": "Drone Detection (Roboflow Universe)",
        "workspace": "brad-dwyer",
        "project": "drone-detection",
        "version": 1,
        "url": "https://universe.roboflow.com/brad-dwyer/drone-detection"
    },
    {
        "name": "UAV Airspace Surveillance",
        "workspace": "drone-dataset",
        "project": "drone-tracking-yolo",
        "version": 2,
        "url": "https://universe.roboflow.com/drone-dataset/drone-tracking-yolo"
    }
]

KAGGLE_POPULAR_DATASETS = [
    {
        "name": "Drone Dataset (UAV)",
        "handle": "dasmehdixtr/drone-dataset-uav",
        "url": "https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav"
    },
    {
        "name": "Anti-UAV Defense Benchmark",
        "handle": "sshikamaru/anti-uav-dataset",
        "url": "https://www.kaggle.com/datasets/sshikamaru/anti-uav-dataset"
    }
]


def download_roboflow_dataset(
    api_key: str,
    workspace: str = "brad-dwyer",
    project: str = "drone-detection",
    version: int = 1,
    target_dir: str = "datasets/drone-dataset"
) -> str:
    """Download dataset directly from Roboflow via Python SDK."""
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=target_dir)
        print(f"[SUCCESS] Roboflow dataset downloaded to {target_dir}")
        return dataset.location
    except Exception as e:
        print(f"[ERROR] Failed to download from Roboflow: {e}")
        raise


def download_kaggle_dataset(dataset_handle: str = "dasmehdixtr/drone-dataset-uav") -> str:
    """Download dataset from Kaggle via kagglehub."""
    try:
        import kagglehub
        path = kagglehub.dataset_download(dataset_handle)
        print(f"[SUCCESS] Kaggle dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[ERROR] Kaggle download failed: {e}")
        raise


def generate_synthetic_drone_dataset(
    output_dir: str = "datasets/synthetic_drone_dataset",
    num_train: int = 30,
    num_val: int = 10,
    num_test: int = 10,
    img_width: int = 640,
    img_height: int = 640
) -> str:
    """
    Generates a full YOLOv8-compliant synthetic dataset with realistic
    drone imagery and normalized YOLO bounding box label txt files.
    """
    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }

    print(f"[INFO] Generating synthetic drone dataset at '{output_dir}'...")

    for split, count in splits.items():
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(count):
            num_drones = random.randint(1, 3) if random.random() > 0.15 else 0
            cam = SimulatedDroneCamera(width=img_width, height=img_height, num_drones=num_drones)
            frame_rgb, gt_boxes = cam.generate_frame()

            filename = f"drone_{split}_{i:04d}"
            img_path = os.path.join(img_dir, f"{filename}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{filename}.txt")

            # Save JPEG (OpenCV expects BGR)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, frame_bgr)

            # Write YOLO formatted annotations: <class-index> <x_center> <y_center> <width> <height>
            with open(lbl_path, "w", encoding="utf-8") as f:
                for (x1, y1, x2, y2) in gt_boxes:
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    x_center = (x1 + bw / 2.0) / img_width
                    y_center = (y1 + bh / 2.0) / img_height
                    norm_w = bw / img_width
                    norm_h = bh / img_height

                    # Class index 0: drone
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    # Generate data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_out = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""# Synthetic Drone Detection Dataset
path: {abs_out}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: drone
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[SUCCESS] Synthetic dataset ready at: {output_dir}")
    print(f"[INFO] Dataset config written to: {yaml_path}")
    return yaml_path


def populate_sample_assets():
    """Populates sample_data/ with sample images and labels for instant app test."""
    os.makedirs("sample_data/sample_images", exist_ok=True)
    os.makedirs("sample_data/sample_labels", exist_ok=True)

    samples = [
        ("sample_drone_single.jpg", 1),
        ("sample_drone_swarm.jpg", 3),
        ("sample_airspace_clear.jpg", 0)
    ]

    for name, num_drones in samples:
        cam = SimulatedDroneCamera(width=854, height=480, num_drones=num_drones)
        frame_rgb, gt_boxes = cam.generate_frame()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img_path = os.path.join("sample_data/sample_images", name)
        cv2.imwrite(img_path, frame_bgr)

        # Write label if drones exist
        base = os.path.splitext(name)[0]
        lbl_path = os.path.join("sample_data/sample_labels", f"{base}.txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            for (x1, y1, x2, y2) in gt_boxes:
                bw = x2 - x1
                bh = y2 - y1
                xc = (x1 + bw / 2.0) / 854.0
                yc = (y1 + bh / 2.0) / 480.0
                nw = bw / 854.0
                nh = bh / 480.0
                f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

    print("[INFO] Sample assets generated in sample_data/")


if __name__ == "__main__":
    populate_sample_assets()
    generate_synthetic_drone_dataset()
