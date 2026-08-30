import json
from pathlib import Path
from PIL import Image

# ==========================================
# CẤU HÌNH
# ==========================================

# File COCO JSON tải từ MakeSense
COCO_JSON = "/content/annotations.json"

# Thư mục chứa ảnh
IMAGE_DIR = "/content/images"

# Thư mục lưu YOLO labels
LABEL_DIR = "/content/labels"

# ==========================================
# TẠO THƯ MỤC
# ==========================================

Path(LABEL_DIR).mkdir(parents=True, exist_ok=True)

# ==========================================
# ĐỌC COCO JSON
# ==========================================

with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# ==========================================
# MAP CATEGORY ID → YOLO CLASS ID
# ==========================================

# COCO có thể có category_id không bắt đầu từ 0
category_id_to_yolo = {
    category["id"]: index
    for index, category in enumerate(categories)
}

print("Classes:")

for category in categories:
    print(
        category["id"],
        "->",
        category_id_to_yolo[category["id"]],
        category["name"]
    )

# ==========================================
# GROUP ANNOTATIONS THEO IMAGE
# ==========================================

annotations_by_image = {}

for ann in annotations:
    image_id = ann["image_id"]

    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(ann)

# ==========================================
# CONVERT
# ==========================================

total_images = 0
total_annotations = 0

for image_info in images:

    image_id = image_info["id"]
    file_name = image_info["file_name"]

    image_path = Path(IMAGE_DIR) / file_name

    # Lấy kích thước ảnh
    if image_path.exists():
        with Image.open(image_path) as img:
            width, height = img.size
    else:
        # Dùng kích thước từ COCO nếu không tìm thấy ảnh
        width = image_info["width"]
        height = image_info["height"]

    # File txt tương ứng
    txt_name = Path(file_name).stem + ".txt"
    txt_path = Path(LABEL_DIR) / txt_name

    lines = []

    # Lấy annotation của ảnh
    anns = annotations_by_image.get(image_id, [])

    for ann in anns:

        category_id = ann["category_id"]

        # ==========================
        # SEGMENTATION
        # ==========================

        segmentation = ann.get("segmentation")

        if not segmentation:
            continue

        # Polygon có thể có nhiều polygon
        if isinstance(segmentation[0], list):
            polygons = segmentation
        else:
            polygons = [segmentation]

        for polygon in polygons:

            # Polygon phải có ít nhất 3 điểm
            if len(polygon) < 6:
                continue

            # COCO:
            # [x1, y1, x2, y2, x3, y3, ...]

            points = []

            for i in range(0, len(polygon), 2):

                x = polygon[i]
                y = polygon[i + 1]

                # Normalize về 0-1
                x = x / width
                y = y / height

                # Giới hạn 0-1
                x = max(0, min(1, x))
                y = max(0, min(1, y))

                points.extend([x, y])

            # YOLO format
            class_id = category_id_to_yolo[category_id]

            line = str(class_id) + " " + " ".join(
                f"{p:.6f}" for p in points
            )

            lines.append(line)

            total_annotations += 1

    # ==========================
    # GHI FILE TXT
    # ==========================

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    total_images += 1

# ==========================================
# KẾT QUẢ
# ==========================================

print("\n================================")
print("CONVERT HOÀN TẤT")
print("================================")

print("Images:", total_images)
print("Annotations:", total_annotations)
print("Labels:", LABEL_DIR)