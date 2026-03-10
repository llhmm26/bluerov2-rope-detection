"""
BlueOS Vision Extension - Main Entry
-----------------------------------
Responsibilities:
- Receive RTSP camera stream from BlueOS
- Run YOLO object detection
- Run rope detection (OpenCV-based)
- Filter detections
- Provide data for:
    - Web dashboard (Cockpit iframe)

This runs INSIDE BlueOS as a Docker extension.
"""

import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, render_template
import platform

from vision.yolo_detector import YOLODetector
from vision.filters import filter_detections, summarize_scene
from vision.rope_detector import RopeDetector  # stub / optional


# ================== CONFIG ==================

RTSP_URL = os.getenv(
    "RTSP_URL",
    "rtsp://192.168.2.2:8554/video_rtsp_stream_0"
)

MODEL_PATH = os.getenv(
    "YOLO_MODEL",
    r"C:\Users\milha\underwaterMonitoring_fyp\vision-assist-extension\app\models\initial-rope-yolo.pt"
)

FRAME_SKIP = 5     # inference frequency
HTTP_PORT = 8080

# ===== REFINEMENT SWITCH =====
USE_REFINEMENT = False #True = YOLO +HSV+Hough | False = YOLO only

WINDOW_NAME = "BlueOS Vision Dashboard"

rope_enabled = False
yolo_enabled = False

# ================= HSV GLOBAL STATE =================
# Default yellow rope range
DEFAULT_HSV_LOWER = np.array([20, 80, 80])
DEFAULT_HSV_UPPER = np.array([35, 255, 255])

hsv_lower = DEFAULT_HSV_LOWER.copy()
hsv_upper = DEFAULT_HSV_UPPER.copy()

# ================== STARTUP LOG ==================

print("=" * 60)
print("[STARTUP] ROV Vision Extension")
print(f"[CONFIG] RTSP_URL   : {RTSP_URL}")
print(f"[CONFIG] YOLO_MODEL : {MODEL_PATH}")
print(f"[CONFIG] FRAME_SKIP : {FRAME_SKIP}")
print("Running on:", platform.platform())
print("=" * 60)

# ----------- GLOBAL STATE (THREAD SAFE) -----------

frame_lock = threading.Lock()
latest_frame = None
latest_summary = {}
latest_detections = {}

stop_flag = False

# --------------------------------------------------
last_summary = None


# ================== VISUAL OVERLAY HELPERS ==================

def draw_yolo_boxes(frame, boxes):
    """
    Draw YOLO object detections (non-rope).
    """
    for det in boxes:
        x1, y1, x2, y2 = det["bbox"]
        label = det.get("class_name", "object")
        conf = det.get("confidence", 0.0)

        color = (0, 255, 0)  # Green for objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

# ---------------- VIDEO PIPELINE ------------------

def video_loop():
    global latest_frame, latest_summary, latest_detections
    global last_summary
    global hsv_lower, hsv_upper

    print("[VIDEO] Opening RTSP:", RTSP_URL)
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[ERROR] Failed to open RTSP stream")
        return
    
    time.sleep(2.0) #Allow camera auto exposure stabilization
    
    yolo = YOLODetector(MODEL_PATH)
    rope_detector = RopeDetector()

    # DEBUG: print YOLO class names
    print("YOLO classes:", yolo.model.names)
    
    # ===== RUNTIME LOG =====
    print("=" * 60)
    print(f"[MODE] Refinement Enabled: {USE_REFINEMENT}")
    print("[RUNTIME] Components initialized")
    print(f"[RUNTIME] YOLO loaded      : {yolo is not None}")
    print(f"[RUNTIME] Rope detector    : {rope_detector is not None}")
    print(f"[RUNTIME] Frame skip       : {FRAME_SKIP}")
    print("=" * 60)

    frame_idx = 0

    #----- CAMERA WARMUP -----
    warmup_frames = 20
    warmup_count = 0

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        #Skip first few unstable frames
        if warmup_count == 0:
            print("[VIDEO] Warming up camera...")

        if warmup_count < warmup_frames:
            warmup_count += 1
            continue
        else:
            if warmup_count == warmup_frames:
                print("[VIDEO] Warmup complete.")
            warmup_count += 1

        frame_idx += 1

        if frame_idx % FRAME_SKIP == 0:

            # ================= STEP A — YOLO =================
            raw_detections = yolo.detect(frame)

            # Filter detections (rope + obstacles)
            filtered = filter_detections(raw_detections)

            rope_detected = False
            rope_position = None

            # ================= STEP B — ROI-BASED ROPE REFINEMENT =================
            for rope in filtered["rope"]:

                x1, y1, x2, y2 = rope["bbox"]

                center_x = (x1 + x2) // 2

                rope_position = (
                    "left" if center_x < frame.shape[1] * 0.33 else
                    "right" if center_x > frame.shape[1] * 0.66 else
                    "center"
                )

                # ================= REFINEMENT CONTROL =================
                if USE_REFINEMENT:

                    roi = frame[y1:y2, x1:x2]

                    found, lines = rope_detector.detect(
                        roi,
                        hsv_lower,
                        hsv_upper
                    )

                    if found:
                        rope_detected = True

                        # Draw refined red lines
                        for (rx1, ry1, rx2, ry2) in lines:
                            cv2.line(
                                frame,
                                (x1 + rx1, y1 + ry1),
                                (x1 + rx2, y1 + ry2),
                                (0, 0, 255),
                                3
                            )
                    else:
                        rope_detected = False

                else:
                    # YOLO only mode
                    rope_detected = True

                    # Draw vertical red line based on YOLO bounding box
                    center_x = (x1 + x2) // 2

                    cv2.line(
                        frame,
                        (center_x, y1),
                        (center_x, y2),
                        (0, 0, 255),   # Red
                        3
                    )

            # ================= STEP C — DRAW YOLO BOXES =================
            draw_yolo_boxes(frame, filtered["obstacles"])

            # ================= STEP D — UPDATE SUMMARY =================
            latest_summary = {
                "rope_detected": rope_detected,
                "rope_position": rope_position
            }

            latest_detections = filtered
        with frame_lock:
            latest_frame = frame.copy()

    #debug to see whether rope is detected
    print("Rope detections:", filtered["ropes"])

    print(filtered)

    cap.release()
    print("[VIDEO] Stopped")


# ---------------- HTTP SERVER ----------------

app = Flask(__name__)


def generate_mjpeg():
    """MJPEG stream for Cockpit iframe"""
    global latest_frame

    while not stop_flag:
        if latest_frame is None:
            time.sleep(0.05)
            continue

        ret, jpg = cv2.imencode(".jpg", latest_frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() +
            b"\r\n"
        )


@app.route("/video")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/status")
def status():
    return jsonify({
        "summary": latest_summary,
        "detections": {
            "obstacles": len(latest_detections.get("obstacles", [])),
            "ropes": len(latest_detections.get("ropes", [])),
        }
    })


@app.route("/toggle_yolo", methods=["POST"])
def toggle_yolo():
    global yolo_enabled
    yolo_enabled = not yolo_enabled
    print("YOLO enabled:", yolo_enabled)
    return jsonify({"yolo_enabled": yolo_enabled})

@app.route("/update_hsv", methods=["POST"])
def update_hsv():
    global hsv_lower, hsv_upper

    data = request.json

    hsv_lower = np.array([
        int(data["hmin"]),
        int(data["smin"]),
        int(data["vmin"])
    ])

    hsv_upper = np.array([
        int(data["hmax"]),
        int(data["smax"]),
        int(data["vmax"])
    ])

    return jsonify({"status": "updated"})


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sample_pixel", methods=["POST"])
def sample_pixel():
    global latest_frame, hsv_lower, hsv_upper

    if latest_frame is None:
        return jsonify({"error": "No frame available"}), 400

    data = request.json
    x = int(data["x"])
    y = int(data["y"])

    with frame_lock:
        frame_copy = latest_frame.copy()

    h, w = frame_copy.shape[:2]
    hsv = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)

    # Boundary safety
    x = max(2, min(w - 3, x))
    y = max(2, min(h - 3, y))

    hsv = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2HSV)

    # Sample small 5x5 region for stability
    region = hsv[y-2:y+3, x-2:x+3]
    mean_color = np.mean(region.reshape(-1, 3), axis=0)

    h_val, s_val, v_val = mean_color

    # Auto adjust range (tunable margins)
    hsv_lower = np.array([
        max(int(h_val - 10), 0),
        # max(int(s_val - 40), 0), # might be too narrow 
        # max(int(v_val - 40), 0) # might be too narrow 
        50, #fixed lower saturation
        50  #fixed lower brightness
    ])

    hsv_upper = np.array([
        min(int(h_val + 10), 179),
        # min(int(s_val + 40), 255),
        # min(int(v_val + 40), 255)
        255,
        255
    ])

    return jsonify({
        "hmin": int(hsv_lower[0]),
        "smin": int(hsv_lower[1]),
        "vmin": int(hsv_lower[2]),
        "hmax": int(hsv_upper[0]),
        "smax": int(hsv_upper[1]),
        "vmax": int(hsv_upper[2])
    })

@app.route("/reset_hsv", methods=["POST"])
def reset_hsv():
    global hsv_lower, hsv_upper

    hsv_lower = DEFAULT_HSV_LOWER.copy()
    hsv_upper = DEFAULT_HSV_UPPER.copy()

    return jsonify({
        "hmin": int(hsv_lower[0]),
        "smin": int(hsv_lower[1]),
        "vmin": int(hsv_lower[2]),
        "hmax": int(hsv_upper[0]),
        "smax": int(hsv_upper[1]),
        "vmax": int(hsv_upper[2])
    })

# ---------------- MAIN ENTRY ----------------

def main():
    rope_detector = RopeDetector()
    print("[START] BlueOS Vision Extension")

    video_thread = threading.Thread(
        target=video_loop,
        daemon=True
    )
    video_thread.start()

    print(f"[HTTP] Dashboard running on port {HTTP_PORT}")
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)


if __name__ == "__main__":
    main()
