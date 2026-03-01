"""
Detection Filtering Logic
-------------------------
Responsibilities:
- Filter YOLO detections
- Classify obstacles vs rope
- Ignore irrelevant detections (fish, tiny objects)
- Output structured, semantic results

"""

from typing import List, Dict


# ---------------- CONFIG ----------------

# Classes that should NOT be treated as obstacles
EXCLUDED_CLASSES = {
    "fish",
    "human"
}

# Minimum bounding box area (pixels) to be considered relevant
MIN_BBOX_AREA = 2500   # tuned for underwater scenes

# Rope class labels (YOLO + CV rope detector will use these)
ROPE_CLASSES = {
    "rope"
}

# ----------------------------------------


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def filter_detections(
    detections: List[Dict],
) -> Dict[str, List[Dict]]:
    """
    Apply filtering rules to YOLO detections.

    Args:
        detections: Raw YOLO detections

    Returns:
        {
            "obstacles": [...],
            "ropes": [...],
            "ignored": [...]
        }
    """
    obstacles = []
    ropes = []
    ignored = []

    for det in detections:
        cls_name = det.get("class_name", "").lower()
        area = bbox_area(det["bbox"])

        # ---- Rule 1: Ignore fish & irrelevant fauna
        if cls_name in EXCLUDED_CLASSES:
            det["ignore_reason"] = "excluded_class"
            ignored.append(det)
            continue

        # ---- Rule 2: Ignore tiny / distant detections
        if area < MIN_BBOX_AREA:
            det["ignore_reason"] = "too_small"
            ignored.append(det)
            continue

        # ---- Rule 3: Rope classification
        if cls_name in ROPE_CLASSES:
            det["category"] = "rope"
            ropes.append(det)
            continue

        # ---- Rule 4: Everything else is an obstacle
        det["category"] = "obstacle"
        obstacles.append(det)

    return {
        "obstacles": obstacles,
        "ropes": ropes,
        "ignored": ignored,
    }


def summarize_scene(filtered, rope_detected) -> Dict:
    """
    Semantic summary used for TTS and dashboard.
    """
    obstacle_count = len(filtered.get("obstacles", []))
    rope_count = len(filtered.get("ropes", []))

    return {
        "rope_detected": rope_detected or rope_count > 0,
        "obstacle_count": obstacle_count,
        "rope_count": rope_count,
    }