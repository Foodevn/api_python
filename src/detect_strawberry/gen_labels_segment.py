from ultralytics import YOLO
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================

MODEL_PATH = "./runs/segment/train-2/weights/best.pt"

# Thư mục chứa ảnh cần auto-label
IMAGE_DIR = Path("./Dataset/chua_co_label/images_rotten")

# Thư mục lưu file YOLO Segmentation .txt
LABEL_DIR = Path("./Dataset/chua_co_label/labels")
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Ngưỡng confidence
CONF_THRESHOLD = 0.5


# =========================
# LOAD MODEL
# =========================

model = YOLO(MODEL_PATH)

print("Classes của model:")
print(model.names)


# =========================
# DUYỆT ẢNH
# =========================

image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.webp",
]

image_paths = []

for extension in image_extensions:
    image_paths.extend(IMAGE_DIR.glob(extension))

print(f"\nTìm thấy {len(image_paths)} ảnh\n")


# =========================
# PREDICT + TẠO LABEL
# =========================

for image_path in image_paths:

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        device=0,
        verbose=False
    )

    result = results[0]

    # File txt tương ứng
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    detections = []

    # =========================
    # KIỂM TRA MASK
    # =========================

    if result.masks is None:

        print(
            f"{image_path.name}: "
            f"Không phát hiện mask"
        )

        # Tạo file txt rỗng
        label_path.touch()

        continue


    # =========================
    # LẤY SEGMENTATION
    # =========================

    masks = result.masks

    # Polygon normalized
    # Mỗi mask tương ứng với 1 object
    polygons = masks.xyn

    # Class ID nằm trong result.boxes
    class_ids = result.boxes.cls.cpu().numpy()

    # Confidence
    confidences = result.boxes.conf.cpu().numpy()


    # =========================
    # DUYỆT TỪNG OBJECT
    # =========================

    for i, polygon in enumerate(polygons):

        class_id = int(class_ids[i])

        confidence = float(confidences[i])


        # Kiểm tra polygon
        if polygon is None or len(polygon) < 3:
            continue


        # =========================
        # YOLO SEGMENTATION FORMAT
        #
        # class_id x1 y1 x2 y2 x3 y3 ...
        # =========================

        line = str(class_id)

        for x, y in polygon:
            line += f" {x:.6f} {y:.6f}"


        detections.append(line)


    # =========================
    # GHI FILE TXT
    # =========================

    with open(label_path, "w", encoding="utf-8") as f:

        for line in detections:
            f.write(line + "\n")


    # =========================
    # LOG
    # =========================

    print(
        f"{image_path.name}: "
        f"phát hiện {len(detections)} object"
    )


# =========================
# HOÀN THÀNH
# =========================

print("\n=========================")
print("Hoàn thành!")
print(f"Labels được lưu tại:")
print(LABEL_DIR)
print("=========================")