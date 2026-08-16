from ultralytics import YOLO
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================

MODEL_PATH = "./runs/detect/train/weights/best.pt"

# Thư mục chứa ảnh cần detect
IMAGE_DIR = Path("./Dataset/chua_co_label/images_rotten")

# Thư mục lưu file YOLO .txt
LABEL_DIR = Path("./Dataset/chua_co_label/labels")
LABEL_DIR.mkdir(exist_ok=True)

# Ngưỡng confidence
CONF_THRESHOLD = 0.5


# =========================
# LOAD MODEL
# =========================

model = YOLO(MODEL_PATH)

print("Classes của model:")
print(model.names)

# Kiểm tra class
# Ví dụ:
# {0: 'ripe', 1: 'unripe', 2: 'rotten'}


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
# DETECT
# =========================

for image_path in image_paths:

    # Chạy YOLO
    results = model.predict(
    source=str(image_path),
    conf=CONF_THRESHOLD,
    device=0,       # GPU NVIDIA đầu tiên
    verbose=False
    )

    result = results[0]

    # File txt tương ứng với ảnh
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    detections = []

    # =========================
    # LẤY BOUNDING BOX
    # =========================

    for box in result.boxes:

        # Class ID
        class_id = int(box.cls[0])

        # Confidence
        confidence = float(box.conf[0])

        # YOLO format normalized:
        # x_center, y_center, width, height
        x_center, y_center, width, height = box.xywhn[0].tolist()

        # Tạo dòng YOLO
        line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        detections.append(line)

    # =========================
    # GHI FILE TXT
    # =========================

    with open(label_path, "w", encoding="utf-8") as f:

        for line in detections:
            f.write(line + "\n")

    print(
        f"{image_path.name}: "
        f"phát hiện {len(detections)} trái dâu"
    )


print("\nHoàn thành!")
print(f"Labels được lưu tại: {LABEL_DIR}")