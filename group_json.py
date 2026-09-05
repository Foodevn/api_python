import json
from pathlib import Path


# =========================================================
# CẤU HÌNH
# =========================================================

INPUT_DIR = Path("./dataset/group/json")
OUTPUT_FILE = Path("./dataset/group/annotations-1.json")


# =========================================================
# COCO DATASET MỚI
# =========================================================

merged = {
    "info": {
        "description": "Merged COCO Dataset"
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
}


# =========================================================
# BIẾN ID
# =========================================================

new_image_id = 1
new_annotation_id = 1

# Mapping tên category -> ID mới
category_name_to_id = {}

# Theo dõi image để tránh trùng
used_image_names = set()


# =========================================================
# ĐỌC TẤT CẢ JSON
# =========================================================

json_files = list(INPUT_DIR.glob("*.json"))

print(f"🔍 Tìm thấy {len(json_files)} file JSON")


for json_file in json_files:

    print(f"\n📂 Đang xử lý: {json_file.name}")

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

    except Exception as e:
        print(f"❌ Không đọc được {json_file.name}: {e}")
        continue


    # =====================================================
    # CATEGORY
    # =====================================================

    old_category_to_new = {}

    for category in coco.get("categories", []):

        old_id = category["id"]
        category_name = category["name"]

        if category_name not in category_name_to_id:

            new_category_id = len(category_name_to_id) + 1

            category_name_to_id[category_name] = new_category_id

            merged["categories"].append({
                "id": new_category_id,
                "name": category_name,
                "supercategory": category.get(
                    "supercategory",
                    ""
                )
            })

        else:
            new_category_id = category_name_to_id[category_name]

        old_category_to_new[old_id] = new_category_id


    # =====================================================
    # IMAGE ID MAPPING
    # =====================================================

    old_image_to_new = {}

    for image in coco.get("images", []):

        old_id = image["id"]
        file_name = image["file_name"]

        # Nếu muốn giữ tất cả ảnh kể cả trùng tên
        # thì bỏ phần kiểm tra này.
        if file_name in used_image_names:

            print(
                f"⚠️ Bỏ ảnh trùng tên: {file_name}"
            )

            continue

        used_image_names.add(file_name)

        old_image_to_new[old_id] = new_image_id

        new_image = image.copy()
        new_image["id"] = new_image_id

        merged["images"].append(new_image)

        new_image_id += 1


    # =====================================================
    # ANNOTATIONS
    # =====================================================

    for annotation in coco.get("annotations", []):

        old_image_id = annotation["image_id"]

        # Annotation thuộc ảnh bị bỏ vì trùng
        if old_image_id not in old_image_to_new:
            continue

        new_annotation = annotation.copy()

        # Đổi image_id
        new_annotation["image_id"] = (
            old_image_to_new[old_image_id]
        )

        # Đổi category_id
        old_category_id = annotation["category_id"]

        if old_category_id not in old_category_to_new:
            print(
                f"⚠️ Không tìm thấy category "
                f"{old_category_id}"
            )
            continue

        new_annotation["category_id"] = (
            old_category_to_new[old_category_id]
        )

        # Đổi annotation ID
        new_annotation["id"] = new_annotation_id

        merged["annotations"].append(
            new_annotation
        )

        new_annotation_id += 1


# =========================================================
# GHI FILE
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        merged,
        f,
        ensure_ascii=False,
        indent=2
    )


# =========================================================
# THỐNG KÊ
# =========================================================

print("\n" + "=" * 50)

print("✅ GỘP DATASET THÀNH CÔNG")

print(f"📸 Images      : {len(merged['images'])}")
print(f"🏷️ Annotations : {len(merged['annotations'])}")
print(f"📦 Categories  : {len(merged['categories'])}")

print("\n📋 Categories:")

for category in merged["categories"]:
    print(
        f"   {category['id']} -> "
        f"{category['name']}"
    )

print(f"\n💾 Output: {OUTPUT_FILE}")