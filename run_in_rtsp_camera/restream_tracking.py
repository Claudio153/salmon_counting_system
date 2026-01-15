import os
import cv2
import subprocess
from ultralytics import YOLO

# ==================================================
# RTSP INPUT CONFIG
# ==================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

MODEL_PATH = "../runs/detect/train/weights/best.pt"  #Reemplazar segun la ruta del modelo entrenado (best.pt)

RTSP_INPUT = (
    "rtsp://user:password@IPcam:554/"         #Reemplazar 'user' 'password' 'IPcam' segun sus datos
    "cam/realmonitor?channel=1&subtype=1"
)

# ==================================================
# RTSP OUTPUT CONFIG (STREAM PROCESADO)
# ==================================================
RTSP_OUTPUT_URL = "rtsp://127.0.0.1:8554/salmon"

STREAM_FPS = 10
CONF_TH = 0.25
IOU_TH = 0.6

COUNT_LINE_Y = 80
SALMON_CLASS_ID = 0

# ==================================================
# CARGAR MODELO
# ==================================================
model = YOLO(MODEL_PATH)

# ==================================================
# VIDEO INPUT (RTSP)
# ==================================================
cap = cv2.VideoCapture(RTSP_INPUT, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir RTSP de entrada")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"RTSP INPUT OK → {width}x{height} @ {STREAM_FPS} FPS")

# ==================================================
# FFmpeg: RTSP OUTPUT (stdin)
# ==================================================
ffmpeg_cmd = [
    "ffmpeg",
    "-loglevel", "error",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{width}x{height}",
    "-r", str(STREAM_FPS),
    "-i", "-",
    "-an",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    RTSP_OUTPUT_URL
]

ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

print("===================================")
print(" RTSP PROCESADO EMITIENDO EN VIVO ")
print(f" {RTSP_OUTPUT_URL}")
print("===================================")

# ==================================================
# VARIABLES DE CONTEO
# ==================================================
counted_ids = set()
prev_positions = {}
total_count = 0

# ==================================================
# LOOP PRINCIPAL
# ==================================================
while True:
    ret, frame = cap.read()
    cv2.putText(
        frame,
        "IA Activa",
        (570, height - 450 ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    

    if not ret:
        print("Frame perdido. Reintentando RTSP...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_INPUT, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue
#-------------------------------------------------------#
    
#------------------------------------------------------------#
    # ---------------------------
    # 1. DETECCIÓN PURA
    # ---------------------------
    det_results = model(
        frame,
        conf=CONF_TH,
        iou=IOU_TH,
        verbose=False
    )

    # ---------------------------
    # 2. TRACKING (solo IDs)
    # ---------------------------
    track_results = model.track(
        frame,
        conf=CONF_TH,
        iou=IOU_TH,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    if (
        not det_results
        or not track_results
        or det_results[0].boxes is None
        or track_results[0].boxes.id is None
    ):
        ffmpeg_proc.stdin.write(frame.tobytes())
        continue

    det_boxes = det_results[0].boxes.xyxy.cpu().numpy()
    det_clss = det_results[0].boxes.cls.cpu().numpy()
    track_ids = track_results[0].boxes.id.cpu().numpy()

    min_len = min(len(det_boxes), len(track_ids))

    for i in range(min_len):
        box = det_boxes[i]
        cls = int(det_clss[i])
        track_id = int(track_ids[i])

        if cls != SALMON_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box)
        center_y = int((y1 + y2) / 2)

        prev_y = prev_positions.get(track_id, center_y)
        prev_positions[track_id] = center_y

        if (
            prev_y >= COUNT_LINE_Y
            and center_y < COUNT_LINE_Y
            and track_id not in counted_ids
        ):
            counted_ids.add(track_id)
            total_count += 1
            print(f"Salmón contado → Total: {total_count}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    # ---------------------------
    # LÍNEA Y CONTADOR
    # ---------------------------
    cv2.putText(
        frame,
        f"Total: {total_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    cv2.line(
            frame,
            (0, COUNT_LINE_Y),
            (width, COUNT_LINE_Y),
            (0, 255, 0),
            2
    )  

    
    # ---------------------------
    # EMITIR FRAME POR RTSP
    # ---------------------------
    ffmpeg_proc.stdin.write(frame.tobytes())
