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