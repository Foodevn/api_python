import json
from pathlib import Path
from inference_sdk import InferenceHTTPClient


# =========================
# CONFIG
# =========================

INPUT_DIR = Path(
    r"./Dataset/chua_co_label/images"
)

OUTPUT_DIR = Path(
    r"./Dataset/chua_co_label/labels_json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# ROBOFLOW CLIENT
# =========================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY"
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
# GET IMAGES
# =========================

images = [
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Tìm thấy {len(images)} ảnh")


# =========================
# PROCESS
# =========================

processed = 0
skipped = 0
failed = 0


for index, image_path in enumerate(images, start=1):

    output_file = OUTPUT_DIR / f"{image_path.stem}.json"

    print(
        f"\n[{index}/{len(images)}] "
        f"{image_path.name}"
    )

    # ==========================================
    # KIỂM TRA JSON ĐÃ TỒN TẠI
    # ==========================================

    if output_file.exists():

        print(
            f"  → Đã có JSON: {output_file.name}"
        )

        print("  → Bỏ qua, không gọi Roboflow")

        skipped += 1

        continue

    # ==========================================
    # CHƯA CÓ JSON → GỌI ROBOFLOW
    # ==========================================

    print("  → Chưa có JSON, đang gọi Roboflow...")

    try:

        result = client.run_workflow(

            workspace_name="hoang-phuc-hzceu",

            workflow_id="general-segmentation-api-6",

            images={
                "image": str(image_path)
            },

            parameters={
                "classes": "Strawberry"
            },

            use_cache=True
        )

        # ==========================================
        # SAVE JSON
        # ==========================================

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"  ✓ Đã lưu: {output_file.name}"
        )

        processed += 1

    except Exception as e:

        print(
            f"  ✗ Lỗi: {e}"
        )

        failed += 1


# =========================
# SUMMARY
# =========================

print("\n================================")
print("Hoàn thành!")
print("================================")

print(f"Đã tải mới   : {processed}")
print(f"Đã bỏ qua    : {skipped}")
print(f"Lỗi          : {failed}")
print(f"Tổng ảnh     : {len(images)}")

print("================================")