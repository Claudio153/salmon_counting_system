from ultralytics import YOLO

# 1. Cargar modelo base
model = YOLO("yolov8n.pt")   #ir probando otros modelos (yolov8s, yolov10-s, yolov10-m)

# 2. Entrenamiento
model.train(
    data="salmon_dataset/data.yaml",
    epochs=120,         
    imgsz=640,
    batch=8,          #Ajustar si la RAM/GPU lo permite 
    patience=25,
    device="cpu",          # cambiar a 'cpu' en caso de 
    workers=4,

    optimizer="AdamW",
    lr0=5e-4,
    cos_lr=True,        #permite que el lr baje suavemente con el tiempo
    
    #Data Augmentation
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=0.0,
    translate=0.03,
    scale=0.4,
    shear=0.0,
    fliplr=0.0,        # NO girar salmones horizontalmente
    mosaic=0.2,
    mixup=0.0,

    close_mosaic=10,
)
