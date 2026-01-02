import hashlib

def bbox_to_id_xyxy(x1, y1, x2, y2, *, decimals=0, as_int=False):
    """
    Create an ID from a bounding box (xyxy).
    - decimals=0 makes the ID stable against tiny float noise (rounding to pixels).
    - as_int=True returns a deterministic 32-bit int derived from the bbox string.
    """
    x1 = round(float(x1), decimals)
    y1 = round(float(y1), decimals)
    x2 = round(float(x2), decimals)
    y2 = round(float(y2), decimals)

    s = f"{x1}_{y1}_{x2}_{y2}"  # bbox-as-ID (string form)

    if not as_int:
        return s

    # Deterministic integer ID from bbox string (stable across runs)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit


def add_bbox_ids_to_detections(detections, *, decimals=0, as_int=False, key="bbox_id"):
    """
    Attaches bbox-derived IDs to `detections` in a best-effort way.
    Supports common patterns:
      - detections.xyxy  (Nx4)
      - detections.data dict (we store ids into detections.data[key])
    Returns the list of IDs (len == number of detections).
    """
    # Get boxes (xyxy). Adapt this if your container uses a different attribute.
    boxes = getattr(detections, "xyxy", None)
    if boxes is None:
        raise AttributeError("detections has no attribute 'xyxy'. Update add_bbox_ids_to_detections to match your detection container.")

    ids = [bbox_to_id_xyxy(*box, decimals=decimals, as_int=as_int) for box in boxes]

    # Store into detections in a common place if available
    if hasattr(detections, "data") and isinstance(detections.data, dict):
        detections.data[key] = ids
    else:
        # fallback: attach attribute directly
        setattr(detections, key, ids)

    return ids


def add_bbox_ids_to_detections(detections, *, decimals=0):
    """
    Add bbox-based IDs directly into `detections`.

    ID format: "x1_y1_x2_y2"
    Stored as:
      - detections.data["id"]  (if available)
      - otherwise detections.id
    """

    if not hasattr(detections, "xyxy"):
        raise AttributeError("detections must have .xyxy (Nx4 bounding boxes)")

    ids = []
    for x1, y1, x2, y2 in detections.xyxy:
        x1 = round(float(x1), decimals)
        y1 = round(float(y1), decimals)
        x2 = round(float(x2), decimals)
        y2 = round(float(y2), decimals)
        ids.append(f"{x1}_{y1}_{x2}_{y2}")

    # Preferred: store in detections.data
    if hasattr(detections, "data") and isinstance(detections.data, dict):
        detections.data["id"] = ids
    else:
        detections.id = ids

    return detections

# ---------------------------
# Example integration snippet
# ---------------------------
# After inference (and after offset/filter if you want IDs in full-frame coords):
#
# detections = model.predict(...)
# if x_offset != 0 or y_offset != 0:
#     detections = offset_detections(detections, x_offset, y_offset, original_h, original_w)
# if scaled_roi and scaled_roi.is_complete():
#     detections = scaled_roi.filter_detections(detections)
#
# Now add IDs from final bbox coordinates:
#
# bbox_ids = add_bbox_ids_to_detections(detections, decimals=0, as_int=False)  # string IDs like "12_44_201_300"
# # or numeric:
# # bbox_ids = add_bbox_ids_to_detections(detections, decimals=0, as_int=True)
#
# Then pass bbox_ids to your drawing routine (recommended):
# annotated = visualizer.draw_detections(frame_resized, detections, show_masks=show_masks, ids=bbox_ids)
#
# Or, if your visualizer cannot accept ids, you can overlay them directly with OpenCV:
#
# for (x1, y1, x2, y2), det_id in zip(detections.xyxy, bbox_ids):
#     x1, y1 = int(x1), int(y1)
#     cv2.putText(frame_resized, f"ID:{det_id}", (x1, max(0, y1 - 6)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
