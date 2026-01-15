import cv2
from ultralytics import YOLO

# ===============================
# CONFIGURACIÓN
# ===============================
MODEL_PATH = "../runs/detect/train/weights/best.pt"
VIDEO_SOURCE = "videos_originales/grabacion_5min.mp4"  # Ruta del video de entrada  
OUTPUT_VIDEO = "videos_con_tracking/output_grabacion5min.mp4" # Ruta del video de salida

CONF_TH = 0.25
IOU_TH = 0.6

# Línea horizontal de conteo (abajo → arriba)
COUNT_LINE_Y = 320

# Clase a contar
SALMON_CLASS_ID = 0

# ===============================
# CARGAR MODELO
# ===============================
model = YOLO(MODEL_PATH)

# ===============================
# VIDEO INPUT
# ===============================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video o stream")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ===============================
# VIDEO OUTPUT
# ===============================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# ===============================
# VARIABLES DE CONTEO
# ===============================
counted_ids = set()
prev_positions = {}
total_count = 0

# ===============================
# LOOP PRINCIPAL
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break
#---------------------------------------------------------------------------        

#--------------------------------------------------------------------------------

    # ---------------------------
    # 1. DETECCIÓN PURA (cajas precisas)
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

    # Validaciones mínimas
    if (
        not det_results
        or not track_results
        or det_results[0].boxes is None
        or track_results[0].boxes.id is None
    ):
        out.write(frame)
        continue

    det_boxes = det_results[0].boxes.xyxy.cpu().numpy()
    det_clss = det_results[0].boxes.cls.cpu().numpy()

    track_ids = track_results[0].boxes.id.cpu().numpy()

    # Asegurar mismo largo (Ultralytics mantiene orden)
    min_len = min(len(det_boxes), len(track_ids))

    # ---------------------------
    # PROCESAR DETECCIONES + IDS
    # ---------------------------
    for i in range(min_len):
        box = det_boxes[i]
        cls = int(det_clss[i])
        track_id = int(track_ids[i])

        if cls != SALMON_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box)
        center_y = int((y1 + y2) / 2)

        # ---------------------------
        # CONTEO (abajo → arriba)
        # ---------------------------
        prev_y = prev_positions.get(track_id, center_y)
        prev_positions[track_id] = center_y

        if (
            prev_y >= COUNT_LINE_Y
            and center_y < COUNT_LINE_Y
            and track_id not in counted_ids
        ):
            counted_ids.add(track_id)
            total_count += 1

        # ---------------------------
        # DIBUJO (CAJA DEL DETECTOR)
        # ---------------------------
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
    cv2.line(
        frame,
        (0, COUNT_LINE_Y),
        (width, COUNT_LINE_Y),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Total: {total_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    out.write(frame)

# ===============================
# CIERRE LIMPIO
# ===============================
cap.release()
out.release()

print("===================================")
print("Proceso finalizado")
print(f"Conteo total: {total_count}")
print(f"Video guardado en: {OUTPUT_VIDEO}")
print("===================================")
