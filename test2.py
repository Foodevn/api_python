import cv2
import time
import torch
from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = r"./runs/runs_1/strawberry_seg/weights/best.pt"

CONF = 0.25
IMGSZ = 640

# Webcam mặc định
CAMERA_ID = 0


# =========================================================
# CHECK GPU
# =========================================================

if torch.cuda.is_available():

    DEVICE = 0

    print("================================")
    print("GPU được sử dụng:")
    print(torch.cuda.get_device_name(0))
    print("================================")

    USE_HALF = True

else:

    DEVICE = "cpu"
    USE_HALF = False

    print("================================")
    print("⚠ Không tìm thấy CUDA")
    print("Đang chạy bằng CPU")
    print("================================")


# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# OPEN CAMERA
# =========================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    raise RuntimeError(
        "Không thể mở camera!"
    )


# Có thể thử độ phân giải này
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("Camera đã mở.")
print("Nhấn Q để thoát.")


# =========================================================
# FPS
# =========================================================

prev_time = time.time()

fps = 0


# =========================================================
# REALTIME LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Không đọc được frame.")

        break


    # =====================================================
    # YOLO PREDICT
    # =====================================================

    results = model.predict(

        source=frame,

        imgsz=IMGSZ,

        conf=CONF,

        device=DEVICE,

        half=USE_HALF,

        verbose=False
    )


    result = results[0]


    # =====================================================
    # VẼ BOX + MASK
    # =====================================================

    annotated_frame = result.plot(

        conf=True,

        labels=True,

        boxes=True,

        masks=True
    )


    # =====================================================
    # CALCULATE FPS
    # =====================================================

    current_time = time.time()

    elapsed = current_time - prev_time

    if elapsed > 0:

        fps = 1 / elapsed

    prev_time = current_time


    # =====================================================
    # GPU MEMORY
    # =====================================================

    if torch.cuda.is_available():

        gpu_memory = (
            torch.cuda.memory_allocated(0)
            / 1024**2
        )

        gpu_text = f"GPU: {gpu_memory:.0f} MB"

    else:

        gpu_text = "CPU"


    # =====================================================
    # DISPLAY FPS
    # =====================================================

    cv2.putText(

        annotated_frame,

        f"FPS: {fps:.1f}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2
    )


    cv2.putText(

        annotated_frame,

        gpu_text,

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Strawberry Segmentation - YOLO11n",
        annotated_frame
    )


    # =====================================================
    # EXIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

print("Đã thoát.")