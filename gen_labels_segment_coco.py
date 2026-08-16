from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import json


# =========================================================
# CẤU HÌNH
# =========================================================

MODEL_PATH = "./runs/segment/train-2/weights/best.pt"

IMAGE_DIR = Path("./Dataset/chua_co_label/images_rotten")

OUTPUT_JSON = Path(
    "./Dataset/chua_co_label/annotations.json"
)

CONF_THRESHOLD = 0.5

DEVICE = 0


# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO(MODEL_PATH)

print("Classes:")
print(model.names)


# =========================================================
# COCO STRUCTURE
# =========================================================

coco = {
    "info": {
        "description": "Strawberry Segmentation"
    },

    "licenses": [],

    "images": [],

    "annotations": [],

    "categories": []
}


# =========================================================
# CATEGORIES
# =========================================================

for class_id, class_name in model.names.items():

    coco["categories"].append({
        "id": class_id + 1,
        "name": class_name,
        "supercategory": "strawberry"
    })


# =========================================================
# IMAGE LIST
# =========================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


image_paths = sorted([
    p
    for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in image_extensions
])


print(f"\nTìm thấy {len(image_paths)} ảnh\n")


# =========================================================
# ID
# =========================================================

image_id = 1
annotation_id = 1


# =========================================================
# PROCESS
# =========================================================

for image_path in image_paths:

    print(f"Đang xử lý: {image_path.name}")


    # -----------------------------------------------------
    # Lấy kích thước ảnh
    # -----------------------------------------------------

    with Image.open(image_path) as img:

        width, height = img.size


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
    # YOLO prediction
    # -----------------------------------------------------

    results = model.predict(

        source=str(image_path),

        conf=CONF_THRESHOLD,

        device=DEVICE,

        verbose=False

    )


    result = results[0]


    # -----------------------------------------------------
    # Không có mask
    # -----------------------------------------------------

    if result.masks is None:

        print("  → Không phát hiện object")

        image_id += 1

        continue


    # -----------------------------------------------------
    # MASK POLYGONS
    # -----------------------------------------------------

    polygons = result.masks.xyn


    # -----------------------------------------------------
    # CLASS + CONFIDENCE
    # -----------------------------------------------------

    class_ids = result.boxes.cls.cpu().numpy()

    confidences = result.boxes.conf.cpu().numpy()


    # -----------------------------------------------------
    # Từng object
    # -----------------------------------------------------

    object_count = 0


    for i, polygon in enumerate(polygons):


        # ================================================
        # CLASS
        # ================================================

        class_id = int(class_ids[i])


        # ================================================
        # CONFIDENCE
        # ================================================

        confidence = float(confidences[i])


        # ================================================
        # POLYGON
        # ================================================

        if polygon is None or len(polygon) < 3:

            continue


        # -----------------------------------------------
        # Convert normalized → pixel
        # -----------------------------------------------

        segmentation = []

        points = []

        for x, y in polygon:

            px = float(x * width)

            py = float(y * height)

            points.append((px, py))

            segmentation.append(round(px, 2))

            segmentation.append(round(py, 2))


        # ================================================
        # BOUNDING BOX
        # ================================================

        xs = [p[0] for p in points]

        ys = [p[1] for p in points]


        min_x = min(xs)

        max_x = max(xs)

        min_y = min(ys)

        max_y = max(ys)


        bbox_width = max_x - min_x

        bbox_height = max_y - min_y


        # ================================================
        # AREA
        # ================================================

        area = 0

        for j in range(len(points)):

            x1, y1 = points[j]

            x2, y2 = points[
                (j + 1) % len(points)
            ]

            area += (
                x1 * y2
                -
                x2 * y1
            )


        area = abs(area) / 2


        # ================================================
        # COCO ANNOTATION
        # ================================================

        annotation = {

            "id": annotation_id,

            "image_id": image_id,

            "category_id": class_id + 1,

            "segmentation": [
                segmentation
            ],

            "area": round(area, 2),

            "bbox": [
                round(min_x, 2),
                round(min_y, 2),
                round(bbox_width, 2),
                round(bbox_height, 2)
            ],

            "iscrowd": 0,

            # Confidence của model
            "score": round(confidence, 6)

        }


        coco["annotations"].append(annotation)


        annotation_id += 1

        object_count += 1


    print(
        f"  → Phát hiện {object_count} object"
    )


    image_id += 1


# =========================================================
# SAVE COCO JSON
# =========================================================

OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        coco,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================================================
# DONE
# =========================================================

print("\n===================================")

print("HOÀN THÀNH")

print("===================================")

print(
    f"Số ảnh: "
    f"{len(coco['images'])}"
)

print(
    f"Số annotation: "
    f"{len(coco['annotations'])}"
)

print(
    f"File COCO: "
    f"{OUTPUT_JSON}"
)