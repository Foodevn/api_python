import os

# Thư mục chứa ảnh
folder = r".\Dataset\doi_ten_anh"

# Các định dạng ảnh hỗ trợ
extensions = (".jpg", ".jpeg", ".png", ".webp")

# Lấy danh sách ảnh
files = [
    f for f in os.listdir(folder)
    if f.lower().endswith(extensions)
]

# Sắp xếp để thứ tự ổn định
files.sort()

# Đổi tên
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder, filename)

    # Giữ nguyên đuôi file
    extension = os.path.splitext(filename)[1].lower()

    new_name = f"strawberry_xau_{i:04d}{extension}"
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

print(f"Đã đổi tên {len(files)} ảnh.")