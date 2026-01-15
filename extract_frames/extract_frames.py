
import cv2
import os

def extraer_frames(video_path, fps_deseados):
    # 1. Crear la carpeta de destino
    output_folder = "extracted_frames" 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Carpeta '{output_folder}' creada.")

    # 2. Cargar el video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    # 3. Obtener info del video original
    fps_video_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS originales: {fps_video_original}")
    print(f"Total de frames en video: {total_frames}")

    # 4. Calcular cada cuántos frames extraer uno
    # Si el video es de 30 fps y quieres 1 fps, extraeremos cada 30 frames.
    intervalo = int(fps_video_original / fps_deseados)
    if intervalo < 1: intervalo = 1

    count = 0
    saved_count = 0

    print(f"Extrayendo 1 frame cada {intervalo} frames reales...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break # Fin del video

        # Solo guardar si es el frame que toca según la frecuencia
        if count % intervalo == 0:
            frame_name = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        count += 1

    # 5. Limpieza
    cap.release()
    print(f"Proceso finalizado. Se guardaron {saved_count} frames en '{output_folder}'.")

# --- CONFIGURACIÓN ---
mi_video = "video.mp4" # Reemplaza con el nombre del archivo
frecuencia = 30            # Cuántos frames por segundo quieres extraer

extraer_frames(mi_video, frecuencia)

