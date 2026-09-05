from pathlib import Path
import re

# =========================
# CẤU HÌNH
# =========================

IMAGE_DIR = Path(r"./dataset/xoa_anh_trung")


# True = thực sự xóa
# False = chỉ xem trước, KHÔNG xóa
DELETE = True


# =========================
# HÀM CHUẨN HÓA TÊN
# =========================

def get_original_name(file):
    """
    Ví dụ:

    powdery_mildew_fruit2.rf.75689d123.jpg
        -> powdery_mildew_fruit2

    powdery_mildew_fruit2_jpg.rf.0ade6735.jpg
        -> powdery_mildew_fruit2

    gray_mold1_jpg.rf.xxxxx.jpg
        -> gray_mold1
    """

    name = file.stem

    # Bỏ .rf.<hash>
    name = re.sub(r"\.rf\.[^.]+$", "", name)

    # Bỏ _jpg ở cuối
    name = re.sub(r"_jpg$", "", name, flags=re.IGNORECASE)

    return name


# =========================
# TÌM ẢNH
# =========================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

files = [
    f for f in IMAGE_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in extensions
]

print(f"Tìm thấy {len(files)} ảnh")


# =========================
# GOM NHÓM
# =========================

groups = {}

for file in files:
    original_name = get_original_name(file)

    groups.setdefault(original_name, []).append(file)


# =========================
# XÓA ẢNH TRÙNG
# =========================

deleted = 0
kept = 0

for original_name, file_list in groups.items():

    # Chỉ có 1 ảnh -> giữ nguyên
    if len(file_list) == 1:
        kept += 1
        continue

    print("\n" + "=" * 70)
    print(f"ẢNH TRÙNG: {original_name}")
    
    # Giữ ảnh đầu tiên
    keep_file = file_list[0]

    print(f"GIỮ : {keep_file.name}")

    # Xóa những ảnh còn lại
    for duplicate in file_list[1:]:

        print(f"XÓA : {duplicate.name}")

        if DELETE:
            duplicate.unlink()

        deleted += 1

    kept += 1


# =========================
# KẾT QUẢ
# =========================

print("\n" + "=" * 70)
print("HOÀN TẤT")
print("=" * 70)

print(f"Tổng ảnh ban đầu : {len(files)}")
print(f"Ảnh giữ lại      : {kept}")
print(f"Ảnh đã xóa       : {deleted}")
print(f"Ảnh còn lại      : {kept}")