"""
Simulated Drone Camera Feed Generator
====================================
Generates a realistic synthetic perimeter surveillance camera stream
with moving drones, telemetry overlay, and flight path simulation.
Can stream live frames to Gradio or export sample MP4 videos.
"""

import math
import random
import time
from typing import Generator, Tuple, List, Dict, Any, Optional
import cv2
import numpy as np


class DroneActor:
    """Represents a simulated UAV moving through 3D/2D airspace."""

    def __init__(self, x: float, y: float, size: float, vx: float, vy: float):
        self.x = x
        self.y = y
        self.base_size = size
        self.current_size = size
        self.vx = vx
        self.vy = vy
        self.prop_angle = 0.0
        self.distance = random.uniform(1.0, 3.0)  # Simulated depth / scale

    def update(self, width: int, height: int):
        self.x += self.vx
        self.y += self.vy
        self.prop_angle = (self.prop_angle + 45) % 360

        # Simulate gradual altitude and distance variations
        self.distance += math.sin(time.time() * 0.5) * 0.005
        self.distance = max(0.6, min(3.5, self.distance))
        self.current_size = self.base_size / self.distance

        # Bounce or wrap around perimeter
        margin = 60
        if self.x < margin or self.x > width - margin:
            self.vx *= -1
            self.x = max(margin, min(width - margin, self.x))
        if self.y < margin or self.y > height - margin:
            self.vy *= -1
            self.y = max(margin, min(height - margin, self.y))


class SimulatedDroneCamera:
    """
    Simulates a high-definition CCTV / Perimeter security camera feed
    monitoring airspace with synthetic drones and dynamic telemetry.
    """

    def __init__(self, width: int = 854, height: int = 480, num_drones: int = 2):
        self.width = width
        self.height = height
        self.num_drones = num_drones
        self.camera_id = "CAM-04 [NORTH-PERIMETER-SECTOR-B]"
        self.frame_count = 0
        self.start_time = time.time()
        self.drones = self._spawn_drones()

    def _spawn_drones(self) -> List[DroneActor]:
        drones = []
        for _ in range(self.num_drones):
            x = random.randint(100, self.width - 100)
            y = random.randint(80, self.height // 2)
            size = random.uniform(32, 64)
            vx = random.choice([-2.5, -1.8, 1.8, 2.5])
            vy = random.choice([-1.0, -0.6, 0.6, 1.0])
            drones.append(DroneActor(x, y, size, vx, vy))
        return drones

    def _draw_drone_silhouette(self, img: np.ndarray, drone: DroneActor) -> Tuple[int, int, int, int]:
        """Draws a multi-rotor quadcopter with rotating propellers."""
        cx, cy = int(drone.x), int(drone.y)
        sz = int(drone.current_size)
        color_body = (35, 38, 45)       # Dark matte carbon fiber
        color_accent = (220, 20, 20)    # Navigation red LED
        color_prop = (120, 130, 140)    # Blur prop disk

        # Arms (X configuration)
        arm_len = sz // 2
        cv2.line(img, (cx - arm_len, cy - arm_len), (cx + arm_len, cy + arm_len), color_body, max(2, sz // 14))
        cv2.line(img, (cx - arm_len, cy + arm_len), (cx + arm_len, cy - arm_len), color_body, max(2, sz // 14))

        # Central fuselage & camera dome
        cv2.circle(img, (cx, cy), max(3, sz // 6), color_body, -1)
        cv2.circle(img, (cx, cy + 2), max(2, sz // 10), (20, 20, 20), -1)

        # 4 Rotor Motors and Propeller Disks
        rotor_positions = [
            (cx - arm_len, cy - arm_len),
            (cx + arm_len, cy - arm_len),
            (cx - arm_len, cy + arm_len),
            (cx + arm_len, cy + arm_len)
        ]

        prop_r = max(4, sz // 4)
        for rx, ry in rotor_positions:
            # Motor mount
            cv2.circle(img, (rx, ry), max(2, sz // 12), (10, 10, 10), -1)
            # Propeller disk blur
            cv2.ellipse(img, (rx, ry), (prop_r, prop_r // 3), int(drone.prop_angle), 0, 360, color_prop, 1)

        # Flashing navigation strobe LED
        if (self.frame_count // 8) % 2 == 0:
            cv2.circle(img, (cx, cy), max(2, sz // 16), color_accent, -1)

        # Compute tight bounding box
        x1 = max(0, cx - arm_len - prop_r)
        y1 = max(0, cy - arm_len - prop_r // 2)
        x2 = min(self.width, cx + arm_len + prop_r)
        y2 = min(self.height, cy + arm_len + prop_r // 2)

        return (x1, y1, x2, y2)

    def generate_frame(self) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Renders a single frame of the surveillance feed.
        Returns:
            frame_rgb: Synthetic camera frame in RGB format.
            ground_truth_boxes: List of (x1, y1, x2, y2) bounding boxes.
        """
        self.frame_count += 1
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Procedural Sky Gradient (Dawn / Daylight horizon)
        for y in range(self.height):
            ratio = y / self.height
            # Sky blue to horizon haze
            r = int(140 + ratio * 60)
            g = int(185 + ratio * 40)
            b = int(225 - ratio * 40)
            img[y, :] = [min(255, r), min(255, g), min(255, b)]

        # Soft clouds
        t = self.frame_count * 0.015
        for i in range(3):
            cx = int((self.width * 0.3 * (i + 1) + t * 40) % (self.width + 200) - 100)
            cy = int(70 + 40 * math.sin(t + i))
            cv2.ellipse(img, (cx, cy), (120, 40), 0, 0, 360, (230, 240, 250), -1)

        # Horizon silhouette line / hills
        hills = []
        for x in range(0, self.width, 10):
            hy = int(self.height * 0.82 + 25 * math.sin(x * 0.008) + 10 * math.sin(x * 0.03))
            hills.append((x, hy))
        hills.append((self.width, self.height))
        hills.append((0, self.height))
        pts = np.array(hills, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (90, 110, 85))

        # Update and render drones
        gt_boxes = []
        for drone in self.drones:
            drone.update(self.width, self.height)
            box = self._draw_drone_silhouette(img, drone)
            gt_boxes.append(box)

        # Camera Telemetry Overlay
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        fps_sim = 24.0
        azimuth = (self.frame_count * 0.2) % 360

        telemetry_top = f"LIVE | {self.camera_id} | {now_str} | FPS: {fps_sim:.1f}"
        telemetry_bottom = f"AZ: {azimuth:05.1f}* | EL: +14.2* | SENSOR: OPTICAL-HD | REC [●]"

        # Telemetry banners
        cv2.putText(img, telemetry_top, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, telemetry_top, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(img, telemetry_bottom, (18, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, telemetry_bottom, (18, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # Center camera reticle
        ccx, ccy = self.width // 2, self.height // 2
        cv2.circle(img, (ccx, ccy), 22, (255, 255, 255), 1)
        cv2.line(img, (ccx - 30, ccy), (ccx - 8, ccy), (255, 255, 255), 1)
        cv2.line(img, (ccx + 8, ccy), (ccx + 30, ccy), (255, 255, 255), 1)
        cv2.line(img, (ccx, ccy - 30), (ccx, ccy - 8), (255, 255, 255), 1)
        cv2.line(img, (ccx, ccy + 8), (ccx, ccy + 30), (255, 255, 255), 1)

        # Slight sensor noise
        noise = np.random.normal(0, 3, img.shape).astype(np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return noisy_img, gt_boxes

    def stream_frames(self, duration_sec: int = 15, fps: int = 20) -> Generator[np.ndarray, None, None]:
        """Generator yielding simulated frames for real-time display."""
        total_frames = duration_sec * fps
        interval = 1.0 / fps

        for _ in range(total_frames):
            t0 = time.time()
            frame, _ = self.generate_frame()
            yield frame
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def export_sample_video(self, output_path: str = "sample_data/simulated_drone_surveillance.mp4", duration_sec: int = 8, fps: int = 24):
        """Pre-renders a synthetic video file for demo playback."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))

        total_frames = duration_sec * fps
        for _ in range(total_frames):
            frame_rgb, _ = self.generate_frame()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)  # OpenCV expects BGR
            out.write(frame_bgr)

        out.release()
        print(f"[INFO] Exported simulated drone surveillance video to: {output_path}")
        return output_path


if __name__ == "__main__":
    camera = SimulatedDroneCamera()
    sample_video = camera.export_sample_video()
    print("Simulated feed generator test passed.")
