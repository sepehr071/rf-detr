from ultralytics import YOLO
import numpy as np

# ---------------- Configuration ----------------
MODEL_PATH = r".\runs3\pose\box_keypoint_detection\weights\best.pt"

# ---------------- Utilities ----------------
def calculate_box_angle(keypoints):
    """
    Calculate orientation angle from front-left and front-right keypoints
    """
    front_left = keypoints[0]
    front_right = keypoints[1]
    dx = front_right[0] - front_left[0]
    dy = front_right[1] - front_left[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

# ---------------- Inference Function ----------------
def infer_image(image_path, model_path=MODEL_PATH):
    """
    Run inference on a single image and return checkpoints (keypoints)

    Returns:
        List[Dict] where each dict contains:
            - class_id
            - class_name
            - box_xyxy
            - box_confidence
            - keypoints (Nx2)
            - keypoint_confidences (N)
            - angle
    """
    model = YOLO(model_path) 
    results = model(image_path, verbose=False)

    detections = []

    for result in results:
        if result.keypoints is None or len(result.keypoints) == 0:
            return detections  # empty list

        kpts_xy = result.keypoints.xy.cpu().numpy()
        kpts_conf = result.keypoints.conf.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()
        boxes_cls = result.boxes.cls.cpu().numpy()

        for i in range(len(kpts_xy)):
            keypoints = kpts_xy[i]
            keypoint_conf = kpts_conf[i]

            detection = {
                "class_id": int(boxes_cls[i]),
                "class_name": result.names[int(boxes_cls[i])],
                "box_xyxy": boxes_xyxy[i].tolist(),
                "box_confidence": float(boxes_conf[i]),
                "keypoints": keypoints.tolist(),                 # checkpoints
                "keypoint_confidences": keypoint_conf.tolist(),
                "angle": calculate_box_angle(keypoints)
            }

            detections.append(detection)

    return detections

# ---------------- Example Usage ----------------
# if __name__ == "__main__":
#     image_path = r".\test.jpg"
#     outputs = infer_image(image_path)

#     for i, det in enumerate(outputs):
#         print(f"\nDetection {i+1}")
#         print(det)


# [
#   {
#     "class_id": 0,
#     "class_name": "box",
#     "box_xyxy": [120.3, 85.1, 420.6, 310.9],
#     "box_confidence": 0.92,
#     "keypoints": [
#         [150.2, 120.8],
#         [310.5, 118.3],
#         [145.1, 260.7],
#         [315.4, 262.1]
#     ],
#     "keypoint_confidences": [0.98, 0.97, 0.95, 0.96],
#     "angle": -0.87
#   }
# ]


