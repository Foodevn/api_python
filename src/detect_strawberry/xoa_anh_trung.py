import cv2
import numpy as np
from pathlib import Path


# =========================================================
# CẤU HÌNH
# =========================================================

IMAGE_DIR = Path("./dataset/xoa_anh_trung")

# Ngưỡng phát hiện ảnh giống nhau
# 0  = giống tuyệt đối
# 5-10 = rất giống
# 10-15 = khá giống
HASH_THRESHOLD = 21

# True  -> xóa ảnh trùng
# False -> chỉ báo ảnh trùng, không xóa
DELETE_DUPLICATES = True


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# PHASH
# =========================================================

def calculate_phash(image_path, hash_size=8, highfreq_factor=4):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError("Không thể đọc ảnh")

    size = hash_size * highfreq_factor

    image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    # DCT
    dct = cv2.dct(
        np.float32(image)
    )

    # Lấy vùng tần số thấp
    dct_low = dct[
        :hash_size,
        :hash_size
    ]

    # Không lấy DC coefficient
    median = np.median(
        dct_low[1:, :]
    )

    hash_value = dct_low > median

    return hash_value


# =========================================================
# KHOẢNG CÁCH GIỮA 2 HASH
# =========================================================

def hash_distance(hash1, hash2):

    return np.count_nonzero(
        hash1 != hash2
    )


# =========================================================
# TÌM ẢNH
# =========================================================

images = [
    p
    for p in IMAGE_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
]


print("=" * 60)
print(f"🔍 Tổng số ảnh: {len(images)}")
print("=" * 60)


# =========================================================
# SO SÁNH ẢNH
# =========================================================

unique_images = []
unique_hashes = []

duplicates = []


for image_path in images:

    try:

        current_hash = calculate_phash(
            image_path
        )

    except Exception as e:

        print(
            f"❌ Lỗi đọc: {image_path.name}"
        )

        print(e)

        continue


    is_duplicate = False


    # So với những ảnh đã giữ lại
    for index, old_hash in enumerate(
        unique_hashes
    ):

        distance = hash_distance(
            current_hash,
            old_hash
        )


        if distance <= HASH_THRESHOLD:

            original = unique_images[index]

            duplicates.append(
                (
                    image_path,
                    original,
                    distance
                )
            )

            print("\n⚠️ PHÁT HIỆN ẢNH TRÙNG")

            print(
                f"   Giữ lại : {original.name}"
            )

            print(
                f"   Xóa     : {image_path.name}"
            )

            print(
                f"   Distance: {distance}"
            )

            is_duplicate = True

            break


    # Không trùng
    if not is_duplicate:

        unique_images.append(
            image_path
        )

        unique_hashes.append(
            current_hash
        )


# =========================================================
# XÓA
# =========================================================

print("\n" + "=" * 60)

print(
    f"🔎 Phát hiện {len(duplicates)} ảnh trùng"
)


if DELETE_DUPLICATES:

    deleted = 0

    for duplicate, original, distance in duplicates:

        try:

            duplicate.unlink()

            deleted += 1

        except Exception as e:

            print(
                f"❌ Không thể xóa: "
                f"{duplicate.name}"
            )

            print(e)

    print(
        f"🗑️ Đã xóa: {deleted} ảnh"
    )

else:

    print(
        "ℹ️ Chế độ kiểm tra - chưa xóa ảnh"
    )


# =========================================================
# THỐNG KÊ
# =========================================================

print("\n" + "=" * 60)

print("✅ HOÀN TẤT")

print(
    f"📸 Ban đầu : {len(images)}"
)

print(
    f"🗑️ Trùng   : {len(duplicates)}"
)

print(
    f"📸 Còn lại : "
    f"{len(images) - len(duplicates)}"
)

print("=" * 60)