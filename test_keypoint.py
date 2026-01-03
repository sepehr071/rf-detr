"""
Test Keypoint Detection Model
=============================
Test the keypoint detection model directly on cropped images.

Usage:
    # Test on a single image
    python test_keypoint.py path/to/cropped_image.jpg

    # Test on all images in debug_crops directory
    python test_keypoint.py debug_crops/

    # Test with a different model
    python test_keypoint.py path/to/image.jpg --model checkpoints/best.pt
"""

import os
import sys
import argparse
import numpy as np
from ultralytics import YOLO


def test_keypoint_model(image_path: str, model_path: str = "checkpoints/best.pt"):
    """
    Test keypoint detection on a single image.
    
    Args:
        image_path: Path to cropped image
        model_path: Path to YOLO pose model
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running inference on: {image_path}")
    results = model(image_path, verbose=False)
    
    # Parse results
    detections = []
    
    for result in results:
        print(f"\n📊 Result for {image_path}:")
        
        if result.keypoints is None or len(result.keypoints) == 0:
            print("   ❌ No keypoints detected")
            continue
        
        kpts_xy = result.keypoints.xy.cpu().numpy()
        kpts_conf = result.keypoints.conf.cpu().numpy()
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
            boxes_cls = result.boxes.cls.cpu().numpy()
        else:
            boxes_xyxy = np.array([])
            boxes_conf = np.array([])
            boxes_cls = np.array([])
        
        for i in range(len(kpts_xy)):
            keypoints = kpts_xy[i]
            keypoint_conf = kpts_conf[i]
            
            print(f"\n   Detection {i + 1}:")
            print(f"   ├─ Keypoints: {len(keypoints)} points")
            
            for j, (kp, conf) in enumerate(zip(keypoints, keypoint_conf)):
                kp_name = ["front_left", "front_right", "back_left", "back_right"][j] if j < 4 else f"point_{j}"
                print(f"   │  ├─ {kp_name}: ({kp[0]:.1f}, {kp[1]:.1f}) conf={conf:.2f}")
            
            if len(boxes_xyxy) > i:
                print(f"   ├─ BBox: [{boxes_xyxy[i][0]:.1f}, {boxes_xyxy[i][1]:.1f}, {boxes_xyxy[i][2]:.1f}, {boxes_xyxy[i][3]:.1f}]")
                print(f"   ├─ Box Confidence: {boxes_conf[i]:.2f}")
                print(f"   └─ Class ID: {int(boxes_cls[i])}")
            
            # Calculate angle from first two keypoints (front edge)
            if len(keypoints) >= 2:
                dx = keypoints[1][0] - keypoints[0][0]
                dy = keypoints[1][1] - keypoints[0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                angle_normalized = (angle - 180) % 360
                print(f"   └─ Calculated angle: {angle_normalized:.1f}°")
            
            detection = {
                "keypoints": keypoints.tolist(),
                "keypoint_confidences": keypoint_conf.tolist(),
            }
            
            if len(boxes_xyxy) > i:
                detection.update({
                    "box_xyxy": boxes_xyxy[i].tolist(),
                    "box_confidence": float(boxes_conf[i]),
                    "class_id": int(boxes_cls[i]),
                })
            
            detections.append(detection)
    
    return detections


def test_directory(dir_path: str, model_path: str = "checkpoints/best.pt"):
    """Test keypoint detection on all images in a directory."""
    if not os.path.isdir(dir_path):
        print(f"❌ Directory not found: {dir_path}")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(dir_path) 
              if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not images:
        print(f"❌ No images found in {dir_path}")
        return
    
    print(f"\n🔍 Testing {len(images)} images in {dir_path}\n")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for image_name in sorted(images):
        image_path = os.path.join(dir_path, image_name)
        result = test_keypoint_model(image_path, model_path)
        
        if result and len(result) > 0 and 'keypoints' in result[0] and len(result[0]['keypoints']) >= 4:
            success_count += 1
        else:
            fail_count += 1
        
        print("-" * 60)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Success: {success_count}/{len(images)}")
    print(f"   ❌ Failed: {fail_count}/{len(images)}")


def main():
    parser = argparse.ArgumentParser(description="Test Keypoint Detection Model")
    parser.add_argument("path", help="Path to image file or directory of images")
    parser.add_argument("--model", default="checkpoints/best.pt", help="Path to YOLO pose model")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        test_directory(args.path, args.model)
    else:
        test_keypoint_model(args.path, args.model)


if __name__ == "__main__":
    main()
