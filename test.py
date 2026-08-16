from ultralytics import YOLO
import cv2

# =========================
# LOAD MODEL
# =========================

MODEL_PATH = "./runs/segment/train-2/weights/best.pt"

model = YOLO(MODEL_PATH)

print("Classes:")
print(model.names)


# =========================
# MỞ CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()


# Có thể giảm độ phân giải để realtime nhanh hơn
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================
# REALTIME
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Không đọc được frame!")
        break


    # =========================
    # YOLO SEGMENTATION
    # =========================

    results = model.predict(
        source=frame,
        conf=0.5,
        device=0,          # GPU NVIDIA
        verbose=False
    )


    result = results[0]


    # =========================
    # VẼ KẾT QUẢ
    # =========================

    annotated_frame = result.plot()


    # =========================
    # HIỂN THỊ
    # =========================

    cv2.imshow(
        "Strawberry Segmentation - Realtime",
        annotated_frame
    )


    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# GIẢI PHÓNG
# =========================

cap.release()
cv2.destroyAllWindows()