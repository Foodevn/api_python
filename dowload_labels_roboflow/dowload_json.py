import os
import json
from pathlib import Path
from inference_sdk import InferenceHTTPClient


# =========================
# CONFIG
# =========================

INPUT_DIR = Path(
    r"./Dataset/chua_co_label/images_rotten_50"
)

OUTPUT_DIR = Path(
    r"./Dataset/chua_co_label/labels_json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# ROBOfLOW CLIENT
# =========================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="rqCG4jwekKwLMcKzCRuz"
)


# =========================
# SUPPORTED IMAGE
# =========================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================
# PROCESS IMAGES
# =========================

images = [
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Tìm thấy {len(images)} ảnh")


for index, image_path in enumerate(images, start=1):

    print(f"\n[{index}/{len(images)}] Đang xử lý: {image_path.name}")

    try:

        result = client.run_workflow(
            workspace_name="hoang-phuc-hzceu",
            workflow_id="general-segmentation-api-5",

            images={
                "image": str(image_path)
            },

            parameters={
                "classes": "Angular Leafspot,Anthracnose Fruit Rot,Blossom Blight,Gray Mold,Healthy-Leaf -Strawberry,Healthy-Strawberry,Leaf Spot,Mulch,Powdery Mildew Fruit,Powdery Mildew Leaf,non-edible-Strawberry"
            },

            use_cache=True
        )

        # =========================
        # LƯU RESULT
        # =========================

        output_file = OUTPUT_DIR / f"{image_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"  ✓ Đã lưu: {output_file.name}")

    except Exception as e:

        print(f"  ✗ Lỗi: {e}")


print("\nHoàn thành!")