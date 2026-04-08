from ultralytics import YOLO
import numpy as np
from depth_estimation import estimate_depth

model = YOLO("yolov8n.pt")


def detect_objects_with_depth(frame):

    results = model(frame, verbose=False)

    depth_map = estimate_depth(frame)

    # -----------------------------
    # Normalize depth map (0 → close, 1 → far)
    # -----------------------------
    depth_min = depth_map.min()
    depth_max = depth_map.max()

    norm_depth = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

    h, w = frame.shape[:2]

    objects = []

    for r in results:
        for box in r.boxes:

            cls = int(box.cls)
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -----------------------------
            # Depth estimation for object
            # -----------------------------
            region = norm_depth[y1:y2, x1:x2]

            if region.size == 0:
                continue

            distance = float(np.mean(region))

            # -----------------------------
            # Direction estimation
            # -----------------------------
            center_x = (x1 + x2) // 2

            if center_x < w * 0.33:
                direction = "left"

            elif center_x < w * 0.66:
                direction = "center"

            else:
                direction = "right"

            objects.append({
                "label": label,
                "distance": distance,
                "direction": direction,
                "box": (x1, y1, x2, y2)
            })

    return objects