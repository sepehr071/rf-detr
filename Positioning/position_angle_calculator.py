import os
import json
import cv2
import numpy as np
import math
from shapely.geometry import Point, box
from shapely import affinity
from camera_calibration import FisheyeCalibrator
from config import SRC_POINTS

# ==================================================================================
# CONSTANTS (from positioning_module.py)
# ==================================================================================

# Object dimensions in meters (width, depth)
OBJECT_DIMENSIONS = {
    0: (0.05, 0.05),      # DS_b_330
    1: (0.125, 0.125),    # DS_b_330_4pa
    2: (0.07, 0.07),      # DS_c_500
    3: (0.05, 0.05),      # HN_b_330
    4: (0.125, 0.18),     # HN_b_330_6pa
    6: (0.07, 0.07),      # HN_c_330, HN_c_500
    -1: (0.06, 0.06),     # Other
}

CIRCLE_CLASSES = {0, 2, 3, 6}

ENABLE_COLLISION_RESOLUTION = True
USE_HOMOGRAPHY = True

# Shelf dimensions in meters
SHELF_WIDTH = 0.51   # X axis
SHELF_DEPTH = 0.385  # Y axis

# ==================================================================================
# ANGLE CALCULATION LOGIC (from angle_calculator.py)
# ==================================================================================

def calculate_angle(p1, p2):
    """Calculate angle of vector p1->p2 in degrees."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def normalize_angle_0_360(angle):
    """Normalize angle to 0 to 359."""
    return angle % 360

def calculate_angles_map(input_file, calibration_file):
    """
    Reads the angle input JSON and calculates rotation angles for each product.
    Returns a dictionary: { int(product_id): float(angle) }
    """
    angle_map = {}
    
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file {calibration_file} not found.")
        return angle_map

    calibrator = FisheyeCalibrator(calibration_file)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return angle_map

    with open(input_file, 'r') as f:
        products = json.load(f)

    # Calibration image size
    calib_w, calib_h = calibrator.image_size if calibrator.image_size else (640, 480)
    
    # Input image size (Assumed 1280x960 based on typical data)
    input_w = 1280
    input_h = 960
    
    scale_x = calib_w / input_w
    scale_y = calib_h / input_h

    for prod in products:
        # Parse product ID (handle string or int)
        try:
            prod_id_raw = prod.get('product_id', 'unknown')
            prod_id = int(prod_id_raw)
        except ValueError:
            print(f"Warning: Could not parse product_id '{prod_id_raw}' as integer.")
            continue

        points = prod.get('points', [])
        
        if len(points) != 4:
            # print(f"Warning: Product {prod_id} has {len(points)} points, expected 4.")
            continue

        # Map points to world coordinates
        world_points = {} # index -> (x, y)
        
        for i, pt in enumerate(points):
            px, py = pt
            if px == -1 or py == -1:
                continue
            
            # Scale to calibration resolution
            px_calib = px * scale_x
            py_calib = py * scale_y
            
            # Map to world
            w_coord = calibrator.pixel_to_world(px_calib, py_calib)
            if w_coord:
                world_points[i] = w_coord
        
        if len(world_points) < 2:
            continue

        # Calculate angles from edges
        angles = []
        
        # Edge 1: Front (BL -> BR) [1 -> 2]
        if 1 in world_points and 2 in world_points:
            ang = calculate_angle(world_points[1], world_points[2])
            angles.append(ang) 
            
        # Edge 2: Back (TL -> TR) [0 -> 3]
        if 0 in world_points and 3 in world_points:
            ang = calculate_angle(world_points[0], world_points[3])
            angles.append(ang) 
            
        # Edge 3: Left (TL -> BL) [0 -> 1]
        if 0 in world_points and 1 in world_points:
            ang = calculate_angle(world_points[0], world_points[1])
            angles.append(ang - 90) 
            
        # Edge 4: Right (TR -> BR) [3 -> 2]
        if 3 in world_points and 2 in world_points:
            ang = calculate_angle(world_points[3], world_points[2])
            angles.append(ang - 90) 
            
        final_angle = 0.0
        if angles:
            # Average angles using vector sum
            sin_sum = sum(math.sin(math.radians(a)) for a in angles)
            cos_sum = sum(math.cos(math.radians(a)) for a in angles)
            final_angle = math.degrees(math.atan2(sin_sum, cos_sum))
            
            # Convert to Clockwise 0-359 as per original logic
            final_angle = (final_angle - 180) % 360
            
            # Round to nearest integer
            final_angle = round(final_angle)
            
            angle_map[prod_id] = normalize_angle_0_360(final_angle)
        else:
            # Fallback: try minAreaRect if 3+ points
            if len(world_points) >= 3:
                pts = np.array(list(world_points.values()), dtype=np.float32)
                rect = cv2.minAreaRect(pts)
                # rect[2] is angle
                angle_map[prod_id] = normalize_angle_0_360(rect[2])

    return angle_map

# ==================================================================================
# POSITIONING LOGIC (from positioning_module.py)
# ==================================================================================

def get_homography_matrix(shelf_width, shelf_depth):
    """Computes the homography matrix from SRC_POINTS to shelf coordinates."""
    src = np.array(SRC_POINTS, dtype=np.float32)
    dst = np.array([
        [0, shelf_depth],
        [shelf_width, shelf_depth],
        [shelf_width, 0],
        [0, 0]
    ], dtype=np.float32)
    
    H, _ = cv2.findHomography(src, dst)
    return H

def parse_detection_file(filepath):
    """Parses the detection output file."""
    objects = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            try:
                parts = line.split('|')
                if len(parts) < 2: continue
                    
                obj_id = line_idx + 1
                mask_str = parts[0].strip()
                mask_data = mask_str.split()
                
                if not mask_data: continue

                class_id = int(mask_data[0])
                coords = [float(x) for x in mask_data[1:]]
                points = []
                for i in range(0, len(coords), 2):
                    if i + 1 < len(coords):
                        points.append((coords[i], coords[i+1]))
                
                bbox_data = []
                if len(parts) >= 2:
                    bbox_str = parts[1].strip()
                    try:
                        bbox_data = [float(x) for x in bbox_str.split()]
                    except ValueError:
                        pass

                objects.append({
                    "id": obj_id,
                    "class_id": class_id,
                    "mask_points": points,
                    "bbox": bbox_data
                })
            except Exception as e:
                print(f"Error parsing line {line_idx}: {e}")
            
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        
    return objects

def find_bottom_centroid(mask_points):
    """Finds the bottom centroid (point with max Y) from mask points."""
    if not mask_points: return None
    return max(mask_points, key=lambda p: p[1])

def create_rotated_geometry(x, y, class_id, rotation_degrees=0):
    """
    Creates a Shapely geometry for an object at (x, y) in meters,
    rotated by rotation_degrees.
    """
    width, depth = OBJECT_DIMENSIONS.get(class_id, (0.05, 0.05))
    
    # Create standard unrotated geometry centered at (x, y)
    if class_id in CIRCLE_CLASSES:
        radius = width / 2.0
        geom = Point(x, y).buffer(radius)
        # Circles don't need rotation unless they are ellipses, but buffer makes a circle.
        # If we wanted an ellipse we'd need to scale. Assuming circles for now.
    else:
        minx = x - width / 2.0
        miny = y - depth / 2.0
        maxx = x + width / 2.0
        maxy = y + depth / 2.0
        geom = box(minx, miny, maxx, maxy)
        
        # Apply rotation
        # Shapely rotates CCW. 
        # Our angle is likely CW (image coords). So we rotate by -rotation_degrees.
        if rotation_degrees != 0:
            geom = affinity.rotate(geom, -rotation_degrees, origin='center')
            
    return geom

def resolve_collisions(objects, shelf_width, shelf_depth, max_iterations=100):
    """Resolves collisions between objects and enforces shelf boundaries."""
    
    # Create geometries
    geometries = []
    for obj in objects:
        wx, wy = obj['world_coords']
        rotation = obj.get('rotation', 0)
        geom = create_rotated_geometry(wx, wy, obj['class_id'], rotation)
        geometries.append({'obj': obj, 'geom': geom})
        
    # --- Pre-processing: Fit all objects into the shelf (Scaling & Shifting) ---
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
    
    scale_x = 1.0
    scale_y = 1.0
    
    if span_x > shelf_width: scale_x = shelf_width / span_x
    if span_y > shelf_depth: scale_y = shelf_depth / span_y
        
    if scale_x < 1.0 or scale_y < 1.0:
        print(f"Scaling objects: X={scale_x:.3f}, Y={scale_y:.3f}")
        for item in geometries:
            c = item['geom'].centroid
            new_cx = min_x_all + (c.x - min_x_all) * scale_x
            new_cy = min_y_all + (c.y - min_y_all) * scale_y
            item['geom'] = affinity.translate(item['geom'], xoff=new_cx - c.x, yoff=new_cy - c.y)

    # Shift to fit
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
    
    if min_x_all < 0: shift_x = -min_x_all
    elif max_x_all > shelf_width: shift_x = shelf_width - max_x_all
        
    if min_y_all < 0: shift_y = -min_y_all
    elif max_y_all > shelf_depth: shift_y = shelf_depth - max_y_all
        
    if shift_x != 0 or shift_y != 0:
        for item in geometries:
            item['geom'] = affinity.translate(item['geom'], xoff=shift_x, yoff=shift_y)
            
    for item in geometries:
        item['initial_centroid'] = item['geom'].centroid

    print(f"Starting collision resolution for {len(objects)} objects...")
    
    for iteration in range(max_iterations):
        moved = False
        
        # Check overlaps
        for i in range(len(geometries)):
            for j in range(i + 1, len(geometries)):
                geom1 = geometries[i]['geom']
                geom2 = geometries[j]['geom']
                
                if geom1.intersects(geom2):
                    intersection = geom1.intersection(geom2)
                    if intersection.is_empty: continue
                        
                    c1_init = geometries[i]['initial_centroid']
                    c2_init = geometries[j]['initial_centroid']
                    
                    dx = c1_init.x - c2_init.x
                    dy = c1_init.y - c2_init.y
                    
                    if abs(dy) > 0.01: dx = 0
                    elif abs(dx) > 0.01: dy = 0
                    else:
                        if abs(dx) > abs(dy): dy = 0
                        else: dx = 0
                    
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < 1e-6:
                        c1_curr = geom1.centroid
                        c2_curr = geom2.centroid
                        dx = c1_curr.x - c2_curr.x
                        dy = c1_curr.y - c2_curr.y
                        if abs(dx) > abs(dy): dy = 0
                        else: dx = 0
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            dx = 0; dy = 1.0; dist = 1.0
                        
                    dx /= dist
                    dy /= dist
                    
                    step_size = 0.001 
                    geometries[i]['geom'] = affinity.translate(geom1, xoff=dx*step_size, yoff=dy*step_size)
                    geometries[j]['geom'] = affinity.translate(geom2, xoff=-dx*step_size, yoff=-dy*step_size)
                    moved = True
        
        # Enforce boundaries
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
            
    # Update object coordinates
    for item in geometries:
        centroid = item['geom'].centroid
        item['obj']['world_coords'] = [centroid.x, centroid.y]
        
    return objects

# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

def main():
    # Configuration
    detection_file = 'frame_20251231_081629_000004.txt'
    angle_input_file = 'sample_input_test1.json'
    calibration_file = 'camera_calibration_full.json'
    output_file = 'object_positions_with_angles.json'
    
    # Image dimensions
    CALIB_WIDTH = 640
    CALIB_HEIGHT = 480
    ORIG_WIDTH = 1280
    ORIG_HEIGHT = 960
    
    # 1. Calculate Angles
    print("Calculating angles...")
    angle_map = calculate_angles_map(angle_input_file, calibration_file)
    print(f"Calculated angles for {len(angle_map)} products.")
    
    # 2. Initialize Positioning
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file {calibration_file} not found.")
        return

    calibrator = FisheyeCalibrator(calibration_file)
    
    H_mat = None
    if USE_HOMOGRAPHY:
        H_mat = get_homography_matrix(SHELF_WIDTH, SHELF_DEPTH)
        print("[INFO] Using Homography for perspective removal.")
    
    if not os.path.exists(detection_file):
        print(f"Error: Detection file {detection_file} not found.")
        return
        
    objects = parse_detection_file(detection_file)
    print(f"Found {len(objects)} objects in detection file.")
    
    valid_objects = []
    error_objects = []
    
    # 3. First pass: Calculate initial world coordinates and assign angles
    for obj in objects:
        obj_id = obj['id']
        mask_points = obj['mask_points']
        bbox = obj.get('bbox')
        
        # Determine bottom point
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            bottom_y = y2
            bottom_point_norm = (center_x, bottom_y)
        else:
            bottom_point_norm = find_bottom_centroid(mask_points)
        
        if bottom_point_norm:
            norm_x, norm_y = bottom_point_norm
            
            px_calib = norm_x * CALIB_WIDTH
            py_calib = norm_y * CALIB_HEIGHT
            
            px_orig = norm_x * ORIG_WIDTH
            py_orig = norm_y * ORIG_HEIGHT
            
            world_coords = None
            
            if USE_HOMOGRAPHY and H_mat is not None:
                pt = np.array([[[px_calib, py_calib]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pt, H_mat)
                world_coords = (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))
            else:
                world_coords = calibrator.pixel_to_world(px_calib, py_calib)
            
            if world_coords:
                wx, wy = world_coords
                
                # Get rotation from angle map (default to 0)
                rotation = angle_map.get(obj_id, 0.0)
                
                valid_objects.append({
                    "object_id": obj_id,
                    "class_id": obj['class_id'],
                    "pixel_coords": [px_orig, py_orig],
                    "world_coords": [wx, wy],
                    "rotation": rotation
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
            
    # 4. Resolve collisions
    if valid_objects and ENABLE_COLLISION_RESOLUTION:
        valid_objects = resolve_collisions(valid_objects, SHELF_WIDTH, SHELF_DEPTH)
        
    # 5. Calculate shelf positions and format results
    results = []
    
    for obj in valid_objects:
        wx, wy = obj['world_coords']
        
        shelf_x = 100 - ((wx / SHELF_WIDTH) * 100)
        shelf_y = (wy / SHELF_DEPTH) * 100
        
        shelf_x = max(0.0, min(100.0, shelf_x))
        shelf_y = max(0.0, min(100.0, shelf_y))
        
        obj['shelf_position'] = [shelf_x, shelf_y]
        results.append(obj)
        print(f"Object {obj['object_id']}: World({wx:.3f}, {wy:.3f}) Rot({obj['rotation']:.1f}) -> Shelf({shelf_x:.1f}, {shelf_y:.1f})")
        
    results.extend(error_objects)
            
    # Save results to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
