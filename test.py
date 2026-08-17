import cv2
from ultralytics import YOLO

# ==========================
# LOAD MODEL
# ==========================
model1 = YOLO("./runs/detect/train/weights/best.pt")  # chín/sống/hư
model2 = YOLO("./runs/segment/train-2/weights/best.pt")  # khuyết tật

# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không mở được camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ==========================
    # MODEL 1
    # ==========================
    results1 = model1(frame, conf=0.5)

    for result in results1:

        boxes = result.boxes

        for box in boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = model1.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Vẽ bbox model 1
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{class_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ==========================
            # CROP TRAI DAU
            # ==========================
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # ==========================
            # CHỈ CHẠY MODEL 2 KHI HƯ
            # ==========================
            if class_name.lower() == "hu":

                results2 = model2(crop, conf=0.4)

                for r2 in results2:

                    for defect_box in r2.boxes:

                        defect_cls = int(defect_box.cls[0])
                        defect_conf = float(defect_box.conf[0])

                        defect_name = model2.names[defect_cls]

                        dx1, dy1, dx2, dy2 = map(
                            int,
                            defect_box.xyxy[0]
                        )

                        # Chuyển tọa độ crop -> frame
                        dx1 += x1
                        dx2 += x1
                        dy1 += y1
                        dy2 += y1

                        cv2.rectangle(
                            frame,
                            (dx1, dy1),
                            (dx2, dy2),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"{defect_name} {defect_conf:.2f}",
                            (dx1, dy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2
                        )

    cv2.imshow("Strawberry Inspection", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()