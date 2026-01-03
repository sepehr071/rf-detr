import os
import json
import cv2
import numpy as np
from typing import Dict
from shapely.geometry import Point, box
from shapely import affinity

# Handle imports for both standalone and module usage
try:
    from .camera_calibration import FisheyeCalibrator
    from .config import SRC_POINTS
except ImportError:
    from camera_calibration import FisheyeCalibrator
    from config import SRC_POINTS

# Object dimensions in meters (width, depth)
# Converted from cm to meters
OBJECT_DIMENSIONS = {
    0: (0.05, 0.05),      # DS_b_330
    1: (0.125, 0.125),    # DS_b_330_4pa
    2: (0.07, 0.07),      # DS_c_500
    3: (0.05, 0.05),      # HN_b_330
    4: (0.125, 0.18),     # HN_b_330_6pa
    6: (0.07, 0.07),      # HN_c_330, HN_c_500
    -1: (0.06, 0.06),      # Other
}

CIRCLE_CLASSES = {0, 2, 3, 6}

# Manual rotations for testing (Object ID -> Rotation 0-3)
# 0: 0 deg, 1: 90 deg, 2: 180 deg, 3: 270 deg
MANUAL_ROTATIONS = {
    # Example:
    # 2: 1, 
    # 5: 0
    #9: 1
}

ENABLE_COLLISION_RESOLUTION = True
USE_HOMOGRAPHY = True

def get_homography_matrix(shelf_width, shelf_depth):
    """
    Computes the homography matrix from SRC_POINTS to shelf coordinates.
    """
    # SRC_POINTS from config.py:
    # [210, 286],      # top-left (Back-Left)
    # [430, 286],      # top-right (Back-Right)
    # [540, 478],      # bottom-right (Front-Right)
    # [85, 478]        # bottom-left (Front-Left)
    
    src = np.array(SRC_POINTS, dtype=np.float32)
    
    # Target points in meters
    # Back-Left -> (0, 0)
    # Back-Right -> (shelf_width, 0)
    # Front-Right -> (shelf_width, shelf_depth)
    # Front-Left -> (0, shelf_depth)
    dst = np.array([
        [0, shelf_depth],
        [shelf_width, shelf_depth],
        [shelf_width, 0],
        [0, 0]
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(src, dst)
    return H

def parse_detection_file(filepath):
    """
    Parses the detection output file.
    
    Args:
        filepath: Path to the detection text file.
        
    Returns:
        List of dictionaries, each containing object info and mask points.
    """
    objects = []
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                parts = line.split('|')
                if len(parts) < 2:
                    print(f"Skipping line {line_idx}: insufficient parts")
                    continue
                    
                # Generate Object ID from line number
                obj_id = line_idx + 1
                
                # Parse Class ID and Mask Points from the FIRST part
                mask_str = parts[0].strip()
                mask_data = mask_str.split()
                
                if not mask_data:
                    print(f"Skipping line {line_idx}: no mask data")
                    continue

                class_id = int(mask_data[0])
                
                # The rest are x, y coordinates
                coords = [float(x) for x in mask_data[1:]]
                points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        points.append((coords[i], coords[i+1]))
                
                # Parse Bounding Box from the SECOND part
                bbox_data = []
                if len(parts) >= 2:
                    bbox_str = parts[1].strip()
                    try:
                        bbox_data = [float(x) for x in bbox_str.split()]
                    except ValueError:
                        print(f"Warning: Could not parse bbox on line {line_idx}")

                objects.append({
                    "id": obj_id,
                    "class_id": class_id,
                    "mask_points": points,
                    "bbox": bbox_data
                })
            except ValueError as ve:
                print(f"Error parsing line {line_idx}: {ve}")
                # print(f"Offending data: {parts[1] if len(parts)>1 else 'N/A'}")
            except Exception as e:
                print(f"Unexpected error on line {line_idx}: {e}")
            
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        
    return objects

def find_bottom_centroid(mask_points):
    """
    Finds the bottom centroid (point with max Y) from mask points.
    
    Args:
        mask_points: List of (x, y) normalized coordinates.
        
    Returns:
        (x, y) tuple of the bottom-most point.
    """
    if not mask_points:
        return None
        
    # Find point with maximum Y
    # If multiple points have the same max Y, we could average their X
    # But for now, let's just take the one with max Y. 
    # If we want to be more robust, we can find all points within a small epsilon of max Y and average them.
    
    max_y = -1.0
    bottom_points = []
    
    for x, y in mask_points:
        if y > max_y:
            max_y = y
            bottom_points = [(x, y)]
        elif y == max_y:
            bottom_points.append((x, y))
            
    # If we want strictly the single lowest point, max() on y works.
    # But "centroid" implies a center. 
    # The user said: "Bottom centroid" means the lowest point (max Y) of the segmentation mask.
    # So I will just take the point with the maximum Y.
    
    # Let's use numpy for easier handling if needed, but simple loop is fine.
    # To handle potential noise, maybe finding the point with max Y is enough.
    
    best_point = max(mask_points, key=lambda p: p[1])
    return best_point

def find_bottom_center_bbox(mask_points):
    """
    Finds the bottom center of the bounding box of the mask points.
    """
    if not mask_points:
        return None
        
    xs = [p[0] for p in mask_points]
    ys = [p[1] for p in mask_points]
    
    min_x, max_x = min(xs), max(xs)
    max_y = max(ys)
    
    center_x = (min_x + max_x) / 2.0
    
    return (center_x, max_y)

def create_geometry(x, y, class_id, rotation=0):
    """
    Creates a Shapely geometry for an object at (x, y) in meters.
    """
    width, depth = OBJECT_DIMENSIONS.get(class_id, (0.05, 0.05))
    
    # Swap dimensions if rotated 90 or 270 (1 or 3)
    if rotation % 2 == 1:
        width, depth = depth, width
    
    if class_id in CIRCLE_CLASSES:
        # For circles, we use a Point buffer. 
        # Note: buffer creates a circle with radius = distance.
        # If width/depth are diameter, radius is half.
        radius = width / 2.0
        return Point(x, y).buffer(radius)
    else:
        # For rectangles, create a box centered at x, y
        minx = x - width / 2.0
        miny = y - depth / 2.0
        maxx = x + width / 2.0
        maxy = y + depth / 2.0
        return box(minx, miny, maxx, maxy)

def resolve_collisions(objects, shelf_width, shelf_depth, max_iterations=100):
    """
    Resolves collisions between objects and enforces shelf boundaries.
    
    Args:
        objects: List of dicts with 'world_coords', 'class_id', 'object_id', 'rotation'.
        shelf_width: Shelf width in meters.
        shelf_depth: Shelf depth in meters.
        max_iterations: Maximum number of iterations for collision resolution.
        
    Returns:
        List of objects with updated 'world_coords'.
    """
    # Create geometries for all objects
    geometries = []
    for obj in objects:
        wx, wy = obj['world_coords']
        rotation = obj.get('rotation', 0)
        geom = create_geometry(wx, wy, obj['class_id'], rotation)
        geometries.append({'obj': obj, 'geom': geom})
        
    # --- Pre-processing: Fit all objects into the shelf (Scaling & Shifting) ---
    # This ensures objects are inside boundaries while preserving relative order.
    
    # 1. Calculate current bounds
    min_x_all = float('inf')
    max_x_all = float('-inf')
    min_y_all = float('inf')
    max_y_all = float('-inf')
    
    for item in geometries:
        minx, miny, maxx, maxy = item['geom'].bounds
        min_x_all = min(min_x_all, minx)
        max_x_all = max(max_x_all, maxx)
        min_y_all = min(min_y_all, miny)
        max_y_all = max(max_y_all, maxy)
        
    span_x = max_x_all - min_x_all
    span_y = max_y_all - min_y_all
    
    # 2. Calculate Scale Factors (only shrink, never expand)
    scale_x = 1.0
    scale_y = 1.0
    
    if span_x > shelf_width:
        scale_x = shelf_width / span_x
        # Add a small margin? No, exact fit is fine for now.
        
    if span_y > shelf_depth:
        scale_y = shelf_depth / span_y
        
    # 3. Apply Scaling
    if scale_x < 1.0 or scale_y < 1.0:
        print(f"Scaling objects to fit shelf: Scale X={scale_x:.3f}, Scale Y={scale_y:.3f}")
        for item in geometries:
            # Scale relative to the min point (preserving order)
            # We need to scale the centroid position, but also the geometry?
            # No, geometry size is fixed (product size). We only scale POSITIONS.
            # But if we scale positions, we might cause overlaps if we shrink too much.
            # But the user said "first put all products in the shelf".
            # If we shrink positions, we might create overlaps, but we preserve order.
            # Then collision resolution fixes overlaps.
            
            # Get current centroid
            c = item['geom'].centroid
            
            # Calculate normalized position (0 to 1) within the span
            if span_x > 0:
                rel_x = (c.x - min_x_all) / span_x
            else:
                rel_x = 0.5
                
            if span_y > 0:
                rel_y = (c.y - min_y_all) / span_y
            else:
                rel_y = 0.5
                
            # New position within the scaled span
            # The new span will be span_x * scale_x (which is <= shelf_width)
            # But wait, we are scaling the BOUNDING BOX.
            # If we scale the positions, the objects themselves don't shrink.
            # So the total span might still be larger than shelf if objects are huge.
            # But this is the best we can do to "fit" them.
            
            # Let's scale the centroid positions based on the bounding box scaling.
            # New centroid = min_x_new + rel_x * new_span_x
            # But we don't know min_x_new yet (shifting comes next).
            # Let's just scale the DISTANCE from min_x_all.
            
            new_cx = min_x_all + (c.x - min_x_all) * scale_x
            new_cy = min_y_all + (c.y - min_y_all) * scale_y
            
            # Move geometry to new centroid
            item['geom'] = affinity.translate(item['geom'], xoff=new_cx - c.x, yoff=new_cy - c.y)

    # 4. Calculate Shift (to center or fit inside 0..Width)
    # Recalculate bounds after scaling
    min_x_all = float('inf')
    max_x_all = float('-inf')
    min_y_all = float('inf')
    max_y_all = float('-inf')
    
    for item in geometries:
        minx, miny, maxx, maxy = item['geom'].bounds
        min_x_all = min(min_x_all, minx)
        max_x_all = max(max_x_all, maxx)
        min_y_all = min(min_y_all, miny)
        max_y_all = max(max_y_all, maxy)
        
    shift_x = 0
    shift_y = 0
    
    # If min < 0, shift right.
    if min_x_all < 0:
        shift_x = -min_x_all
    # If max > width, shift left (shouldn't happen if scaled, unless min > 0 and max > width)
    elif max_x_all > shelf_width:
        shift_x = shelf_width - max_x_all
        
    if min_y_all < 0:
        shift_y = -min_y_all
    elif max_y_all > shelf_depth:
        shift_y = shelf_depth - max_y_all
        
    if shift_x != 0 or shift_y != 0:
        print(f"Shifting objects to fit shelf: Shift X={shift_x:.3f}, Shift Y={shift_y:.3f}")
        for item in geometries:
            item['geom'] = affinity.translate(item['geom'], xoff=shift_x, yoff=shift_y)
            
    # Store initial positions after scaling/shifting to preserve relative order
    for item in geometries:
        item['initial_centroid'] = item['geom'].centroid

    print(f"Starting collision resolution for {len(objects)} objects...")
    
    for iteration in range(max_iterations):
        moved = False
        
        # Check for overlaps between objects
        for i in range(len(geometries)):
            for j in range(i + 1, len(geometries)):
                geom1 = geometries[i]['geom']
                geom2 = geometries[j]['geom']
                
                if geom1.intersects(geom2):
                    # Calculate overlap
                    intersection = geom1.intersection(geom2)
                    if intersection.is_empty:
                        continue
                        
                    # Use initial relative direction to preserve order
                    c1_init = geometries[i]['initial_centroid']
                    c2_init = geometries[j]['initial_centroid']
                    
                    dx = c1_init.x - c2_init.x
                    dy = c1_init.y - c2_init.y
                    
                    # Stronger Order Preservation: 
                    # If they were separated by more than 1cm in an axis, 
                    # ONLY push in that axis.
                    if abs(dy) > 0.01:
                        dx = 0
                    elif abs(dx) > 0.01:
                        dy = 0
                    else:
                        # Use dominant axis
                        if abs(dx) > abs(dy): dy = 0
                        else: dx = 0
                    
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < 1e-6:
                        # Fallback to current centroids
                        c1_curr = geom1.centroid
                        c2_curr = geom2.centroid
                        dx = c1_curr.x - c2_curr.x
                        dy = c1_curr.y - c2_curr.y
                        if abs(dx) > abs(dy): dy = 0
                        else: dx = 0
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            dx = 0; dy = 1.0; dist = 1.0
                        
                    # Normalize direction
                    dx /= dist
                    dy /= dist
                    
                    # Move each object by a small step
                    # Use a smaller step for better stability
                    step_size = 0.001 
                    
                    # Move geom1
                    geometries[i]['geom'] = affinity.translate(geom1, xoff=dx*step_size, yoff=dy*step_size)
                    # Move geom2
                    geometries[j]['geom'] = affinity.translate(geom2, xoff=-dx*step_size, yoff=-dy*step_size)
                    
                    moved = True
        
        # Enforce shelf boundaries (Softly)
        for i in range(len(geometries)):
            geom = geometries[i]['geom']
            minx, miny, maxx, maxy = geom.bounds
            
            dx = 0
            dy = 0
            
            if minx < 0: dx = -minx
            elif maxx > shelf_width: dx = shelf_width - maxx
                
            if miny < 0: dy = -miny
            elif maxy > shelf_depth: dy = shelf_depth - maxy
                
            if dx != 0 or dy != 0:
                geometries[i]['geom'] = affinity.translate(geom, xoff=dx, yoff=dy)
                moved = True
                
        if not moved:
            print(f"Collision resolution converged after {iteration} iterations.")
            break
    else:
        print(f"Collision resolution reached max iterations ({max_iterations}).")
            
    # Update object coordinates
    for item in geometries:
        centroid = item['geom'].centroid
        item['obj']['world_coords'] = [centroid.x, centroid.y]
        
    return objects


def calculate_positions_from_detections(
    detections,
    image_width: int = 1280,
    image_height: int = 960,
    calibration_file: str = None,
    enable_collision_resolution: bool = False,
    shelf_width: float = 0.51,
    shelf_depth: float = 0.385,
    calib_width: int = 640,
    calib_height: int = 480,
    use_homography: bool = True,
    angle_map: Dict[str, float] = None,
) -> list:
    """
    Calculate world positions directly from a supervision Detections object.

    Args:
        detections: supervision.Detections object with xyxy, class_id, and optionally mask
        image_width: Image width for coordinate scaling
        image_height: Image height for coordinate scaling
        calibration_file: Path to camera calibration JSON (default: same directory's camera_calibration_full.json)
        enable_collision_resolution: Whether to resolve collisions (default: False)
        shelf_width: Shelf width in meters
        shelf_depth: Shelf depth in meters
        calib_width: Calibration image width
        calib_height: Calibration image height
        use_homography: Use homography transform (True) or ray-plane intersection (False)
        angle_map: Dict mapping bbox_id (x1_y1_x2_y2) to rotation angle (0-359)

    Returns:
        List of position dictionaries with object_id, class_id, pixel_coords, world_coords, shelf_position
    """
    if len(detections) == 0:
        return []

    # Default calibration file path
    if calibration_file is None:
        calibration_file = os.path.join(os.path.dirname(__file__), 'camera_calibration_full.json')

    # Initialize Calibrator
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file {calibration_file} not found.")
        return []

    calibrator = FisheyeCalibrator(calibration_file)

    # Initialize Homography if enabled
    H_mat = None
    if use_homography:
        H_mat = get_homography_matrix(shelf_width, shelf_depth)

    valid_objects = []
    error_objects = []

    # Process each detection
    for obj_id, (box, class_id) in enumerate(zip(detections.xyxy, detections.class_id), start=1):
        x1, y1, x2, y2 = box

        # Get bottom center of bounding box (normalized)
        center_x = (x1 + x2) / 2.0 / image_width
        bottom_y = y2 / image_height

        # Convert to calibration pixel coordinates for world mapping
        px_calib = center_x * calib_width
        py_calib = bottom_y * calib_height

        # Convert to original pixel coordinates for visualization
        px_orig = center_x * image_width
        py_orig = bottom_y * image_height

        # Get world coordinates
        world_coords = None

        if use_homography and H_mat is not None:
            # Apply homography to pixel coordinates
            pt = np.array([[[px_calib, py_calib]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, H_mat)
            world_coords = (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))
        else:
            # Use ray-plane intersection
            world_coords = calibrator.pixel_to_world(px_calib, py_calib)

        if world_coords:
            wx, wy = world_coords

            # Generate bbox_id for angle_map lookup
            bbox_id = f"{int(round(x1))}_{int(round(y1))}_{int(round(x2))}_{int(round(y2))}"

            # Get rotation from angle_map if provided, otherwise fallback to MANUAL_ROTATIONS
            if angle_map and bbox_id in angle_map:
                rotation = angle_map[bbox_id]
            else:
                rotation = MANUAL_ROTATIONS.get(obj_id, 0)

            valid_objects.append({
                "object_id": obj_id,
                "class_id": int(class_id),
                "pixel_coords": [px_orig, py_orig],
                "world_coords": [wx, wy],
                "rotation": rotation,
                "bbox_id": bbox_id,
            })
        else:
            error_objects.append({
                "object_id": obj_id,
                "class_id": int(class_id),
                "error": "Could not compute world coordinates"
            })

    # Resolve collisions if enabled
    if valid_objects and enable_collision_resolution:
        valid_objects = resolve_collisions(valid_objects, shelf_width, shelf_depth)

    # Calculate shelf positions and format results
    results = []

    for obj in valid_objects:
        wx, wy = obj['world_coords']

        # Normalize to 0-100 based on shelf dimensions
        shelf_x = (wx / shelf_width) * 100
        shelf_y = (wy / shelf_depth) * 100

        # Clamp values
        shelf_x = max(0.0, min(100.0, shelf_x))
        shelf_y = max(0.0, min(100.0, shelf_y))

        obj['shelf_position'] = [shelf_x, shelf_y]
        results.append(obj)

    results.extend(error_objects)

    return results


def calculate_positions(
    detection_file: str,
    output_file: str = None,
    calibration_file: str = None,
    enable_collision_resolution: bool = False,
    shelf_width: float = 0.51,
    shelf_depth: float = 0.385,
    image_width: int = 1280,
    image_height: int = 960,
    calib_width: int = 640,
    calib_height: int = 480,
    use_homography: bool = True
) -> list:
    """
    Calculate world positions from detection file.

    Args:
        detection_file: Path to detection .txt file
        output_file: Path to save output JSON (optional)
        calibration_file: Path to camera calibration JSON (default: same directory's camera_calibration_full.json)
        enable_collision_resolution: Whether to resolve collisions (default: False)
        shelf_width: Shelf width in meters
        shelf_depth: Shelf depth in meters
        image_width: Original image width for coordinate scaling
        image_height: Original image height for coordinate scaling
        calib_width: Calibration image width
        calib_height: Calibration image height
        use_homography: Use homography transform (True) or ray-plane intersection (False)

    Returns:
        List of position dictionaries with object_id, class_id, pixel_coords, world_coords, shelf_position
    """
    # Default calibration file path
    if calibration_file is None:
        calibration_file = os.path.join(os.path.dirname(__file__), 'camera_calibration_full.json')

    # Initialize Calibrator
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file {calibration_file} not found.")
        return []

    calibrator = FisheyeCalibrator(calibration_file)

    # Initialize Homography if enabled
    H_mat = None
    if use_homography:
        H_mat = get_homography_matrix(shelf_width, shelf_depth)

    # Parse Detections
    if not os.path.exists(detection_file):
        print(f"Error: Detection file {detection_file} not found.")
        return []

    objects = parse_detection_file(detection_file)

    if not objects:
        return []

    valid_objects = []
    error_objects = []

    # First pass: Calculate initial world coordinates
    for obj in objects:
        obj_id = obj['id']
        mask_points = obj['mask_points']
        bbox = obj.get('bbox')

        # Use BBox from file (Bottom Center) if available
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            bottom_y = y2
            bottom_point_norm = (center_x, bottom_y)
        else:
            bottom_point_norm = find_bottom_centroid(mask_points)

        if bottom_point_norm:
            norm_x, norm_y = bottom_point_norm

            # Convert to calibration pixel coordinates for world mapping
            px_calib = norm_x * calib_width
            py_calib = norm_y * calib_height

            # Convert to original pixel coordinates for visualization
            px_orig = norm_x * image_width
            py_orig = norm_y * image_height

            # Get world coordinates
            world_coords = None

            if use_homography and H_mat is not None:
                # Apply homography to pixel coordinates
                pt = np.array([[[px_calib, py_calib]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pt, H_mat)
                world_coords = (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))
            else:
                # Use ray-plane intersection
                world_coords = calibrator.pixel_to_world(px_calib, py_calib)

            if world_coords:
                wx, wy = world_coords
                valid_objects.append({
                    "object_id": obj_id,
                    "class_id": obj['class_id'],
                    "pixel_coords": [px_orig, py_orig],
                    "world_coords": [wx, wy],
                    "rotation": MANUAL_ROTATIONS.get(obj_id, 0)
                })
            else:
                error_objects.append({
                    "object_id": obj_id,
                    "class_id": obj['class_id'],
                    "error": "Could not compute world coordinates"
                })
        else:
            error_objects.append({
                "object_id": obj_id,
                "class_id": obj['class_id'],
                "error": "No mask points found"
            })

    # Resolve collisions if enabled
    if valid_objects and enable_collision_resolution:
        valid_objects = resolve_collisions(valid_objects, shelf_width, shelf_depth)

    # Second pass: Calculate shelf positions and format results
    results = []

    for obj in valid_objects:
        wx, wy = obj['world_coords']

        # Normalize to 0-100 based on shelf dimensions
        shelf_x = (wx / shelf_width) * 100
        shelf_y = (wy / shelf_depth) * 100

        # Clamp values
        shelf_x = max(0.0, min(100.0, shelf_x))
        shelf_y = max(0.0, min(100.0, shelf_y))

        obj['shelf_position'] = [shelf_x, shelf_y]
        results.append(obj)

    results.extend(error_objects)

    # Save results to JSON if output_file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    return results


def main():
    """Standalone entry point with hardcoded defaults."""
    # Configuration
    detection_file = 'frame_20251229_110832_000006.txt'
    output_file = 'object_positions.json'

    results = calculate_positions(
        detection_file=detection_file,
        output_file=output_file,
        enable_collision_resolution=ENABLE_COLLISION_RESOLUTION,
        use_homography=USE_HOMOGRAPHY
    )

    # Print summary
    valid_count = sum(1 for r in results if 'shelf_position' in r)
    error_count = sum(1 for r in results if 'error' in r)
    print(f"Processed {len(results)} objects: {valid_count} valid, {error_count} errors")

    for obj in results:
        if 'shelf_position' in obj:
            wx, wy = obj['world_coords']
            sx, sy = obj['shelf_position']
            print(f"Object {obj['object_id']}: World({wx:.3f}, {wy:.3f}) -> Shelf({sx:.1f}, {sy:.1f})")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
