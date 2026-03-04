from typing import List, Dict

EXCLUDED_CLASSES = set() #{"fish", "human"}
MIN_BBOX_AREA = 2500


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def filter_detections(detections: List[Dict]):

    obstacles = []
    ropes = []
    ignored = []

    for det in detections:
        cls_name = det.get("class_name", "").lower()
        area = bbox_area(det["bbox"])

        if cls_name in EXCLUDED_CLASSES:
            ignored.append(det)
            continue

        if area < MIN_BBOX_AREA:
            ignored.append(det)
            continue

        if cls_name == "rope":
            ropes.append(det)
        else:
            obstacles.append(det)

    return {
        "obstacles": obstacles,
        "ropes": ropes,
        "ignored": ignored,
    }


def summarize_scene(filtered):

    return {
        "rope_detected": len(filtered.get("ropes", [])) > 0,
        "obstacle_count": len(filtered.get("obstacles", [])),
        "rope_count": len(filtered.get("ropes", [])),
    }