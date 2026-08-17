import json
from pathlib import Path
from PIL import Image


# =========================================================
# CONFIG
# =========================================================

# Thư mục chứa ảnh
IMAGE_DIR = Path(
    r"./Dataset/chua_co_label/images_50"
)

# Thư mục chứa YOLO segmentation TXT
LABEL_DIR = Path(
    r"./Dataset/chua_co_label/labels_txt"
)

# File COCO đầu ra
OUTPUT_JSON = Path(
    r"./Dataset/chua_co_label/annotations.json"
)


# =========================================================
# CLASS
# =========================================================

# Hiện tại Roboflow của bạn chỉ có:
# 0 = strawberry

CLASS_NAMES = {
    0: "strawberry",
    1: "strawberry1"
}


# =========================================================
# COCO STRUCTURE
# =========================================================

coco = {
    "info": {
        "description": "Strawberry Segmentation Dataset"
    },

    "licenses": [],

    "images": [],

    "annotations": [],

    "categories": []
}


# =========================================================
# CATEGORIES
# =========================================================

for class_id, class_name in CLASS_NAMES.items():

    coco["categories"].append({
        "id": class_id + 1,
        "name": class_name,
        "supercategory": "strawberry"
    })


# =========================================================
# FIND IMAGES
# =========================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

image_files.sort()

print(f"Tìm thấy {len(image_files)} ảnh")


# =========================================================
# ID
# =========================================================

image_id = 1
annotation_id = 1

total_annotations = 0


# =========================================================
# PROCESS EACH IMAGE
# =========================================================

for image_path in image_files:

    print(
        f"[{image_id}/{len(image_files)}] "
        f"{image_path.name}"
    )

    # -----------------------------------------------------
    # Tìm TXT tương ứng
    # -----------------------------------------------------

    label_path = LABEL_DIR / (
        image_path.stem + ".txt"
    )

    # -----------------------------------------------------
    # Đọc kích thước ảnh
    # -----------------------------------------------------

    try:

        with Image.open(image_path) as img:

            width, height = img.size

    except Exception as e:

        print(
            f"  ✗ Không đọc được ảnh: {e}"
        )

        image_id += 1
        continue

    # -----------------------------------------------------
    # Thêm image vào COCO
    # -----------------------------------------------------

    coco["images"].append({

        "id": image_id,

        "file_name": image_path.name,

        "width": width,

        "height": height

    })

    # -----------------------------------------------------
    # Không có TXT
    # -----------------------------------------------------

    if not label_path.exists():

        print(
            "  ! Không có file TXT"
        )

        image_id += 1
        continue

    # -----------------------------------------------------
    # Đọc TXT
    # -----------------------------------------------------

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    # =====================================================
    # PROCESS EACH OBJECT
    # =====================================================

    for line in lines:

        line = line.strip()

        if not line:
            continue

        values = line.split()

        # -------------------------------------------------
        # YOLO segmentation cần ít nhất:
        #
        # class x1 y1 x2 y2 x3 y3
        # -------------------------------------------------

        if len(values) < 7:

            print(
                "  ! Bỏ annotation không hợp lệ"
            )

            continue

        try:

            class_id = int(values[0])

            coords = [
                float(x)
                for x in values[1:]
            ]

        except ValueError:

            print(
                "  ! Không đọc được annotation"
            )

            continue

        # -------------------------------------------------
        # Số tọa độ phải là số chẵn
        # -------------------------------------------------

        if len(coords) % 2 != 0:

            print(
                "  ! Số tọa độ không hợp lệ"
            )

            continue

        # -------------------------------------------------
        # Class không tồn tại
        # -------------------------------------------------

        if class_id not in CLASS_NAMES:

            print(
                f"  ! Class {class_id} không tồn tại"
            )

            continue

        # =================================================
        # YOLO NORMALIZED
        #
        # x = 0 -> 1
        # y = 0 -> 1
        #
        # chuyển sang pixel
        # =================================================

        polygon = []

        xs = []
        ys = []

        for i in range(
            0,
            len(coords),
            2
        ):

            x_norm = coords[i]

            y_norm = coords[i + 1]

            # ---------------------------------------------
            # YOLO -> pixel
            # ---------------------------------------------

            x = x_norm * width

            y = y_norm * height

            # ---------------------------------------------
            # Giới hạn trong ảnh
            # ---------------------------------------------

            x = max(
                0,
                min(width, x)
            )

            y = max(
                0,
                min(height, y)
            )

            polygon.append(x)
            polygon.append(y)

            xs.append(x)
            ys.append(y)

        # -------------------------------------------------
        # Cần ít nhất 3 điểm
        # -------------------------------------------------

        if len(polygon) < 6:

            continue

        # =================================================
        # BOUNDING BOX
        # =================================================

        x_min = min(xs)

        y_min = min(ys)

        x_max = max(xs)

        y_max = max(ys)

        bbox_width = x_max - x_min

        bbox_height = y_max - y_min

        # =================================================
        # AREA
        # =================================================

        # Công thức Shoelace
        area = 0.0

        number_of_points = len(polygon) // 2

        for i in range(number_of_points):

            x1 = polygon[
                2 * i
            ]

            y1 = polygon[
                2 * i + 1
            ]

            j = (
                i + 1
            ) % number_of_points

            x2 = polygon[
                2 * j
            ]

            y2 = polygon[
                2 * j + 1
            ]

            area += (
                x1 * y2
                - x2 * y1
            )

        area = abs(area) / 2

        # =================================================
        # COCO ANNOTATION
        # =================================================

        annotation = {

            "id": annotation_id,

            "image_id": image_id,

            # COCO category ID bắt đầu từ 1
            "category_id": class_id + 1,

            "segmentation": [
                polygon
            ],

            "area": area,

            "bbox": [
                x_min,
                y_min,
                bbox_width,
                bbox_height
            ],

            "iscrowd": 0

        }

        coco["annotations"].append(
            annotation
        )

        annotation_id += 1

        total_annotations += 1

    image_id += 1


# =========================================================
# SAVE JSON
# =========================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        coco,
        f,
        ensure_ascii=False,
        indent=2
    )


# =========================================================
# RESULT
# =========================================================

print()
print("==============================")
print("Hoàn thành!")
print(
    f"Images: {len(coco['images'])}"
)
print(
    f"Annotations: {total_annotations}"
)
print(
    f"Categories: {len(coco['categories'])}"
)
print(
    f"Output: {OUTPUT_JSON}"
)
print("==============================")