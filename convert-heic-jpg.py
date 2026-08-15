from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener

# Cho Pillow đọc được HEIC
register_heif_opener()

# Thư mục ảnh HEIC
INPUT_DIR = Path("./Dataset/chua_co_label/images_heic")

# Thư mục chứa ảnh JPG
OUTPUT_DIR = Path("./Dataset/chua_co_label/images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Duyệt tất cả file
for file in INPUT_DIR.iterdir():

    if file.suffix.lower() not in [".heic", ".heif"]:
        continue

    try:
        # Mở ảnh HEIC
        image = Image.open(file)

        # Chuyển sang RGB
        image = image.convert("RGB")

        # Tên file JPG
        output_file = OUTPUT_DIR / f"{file.stem}.jpg"

        # Lưu JPG
        image.save(
            output_file,
            "JPEG",
            quality=95
        )

        print(f"Đã chuyển: {file.name} -> {output_file.name}")

    except Exception as e:
        print(f"Lỗi {file.name}: {e}")

print("\nHoàn thành!")