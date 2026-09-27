from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Chia dataset từ non_split sang fruit_dataset theo train/val"
	)
	parser.add_argument(
		"--source-root",
		type=Path,
		default=Path("Dataset/non_split"),
		help="Thư mục nguồn chứa image/ và label/",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("Dataset/fruit_dataset"),
		help="Thư mục đích fruit_dataset",
	)
	parser.add_argument(
		"--val-ratio",
		type=float,
		default=0.2,
		help="Tỉ lệ dữ liệu cho val (mặc định: 0.2)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Seed để chia dữ liệu reproducible",
	)
	return parser.parse_args()


def ensure_output_dirs(output_root: Path) -> None:
	(output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
	(output_root / "images" / "val").mkdir(parents=True, exist_ok=True)
	(output_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
	(output_root / "labels" / "val").mkdir(parents=True, exist_ok=True)


def clear_split_dirs(output_root: Path) -> None:
	for split_name in ("train", "val"):
		for folder in (
			output_root / "images" / split_name,
			output_root / "labels" / split_name,
		):
			for item in folder.iterdir():
				if item.is_file():
					item.unlink()


def build_image_index(image_dir: Path) -> dict[str, Path]:
	index: dict[str, Path] = {}
	for image_path in image_dir.iterdir():
		if image_path.is_file():
			index[image_path.stem] = image_path
	return index


def collect_pairs(image_dir: Path, label_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
	image_index = build_image_index(image_dir)
	pairs: list[tuple[Path, Path]] = []
	missing_images: list[Path] = []

	for label_path in sorted(label_dir.glob("*.txt")):
		stem = label_path.stem
		image_path = image_index.get(stem)
		if image_path is None:
			missing_images.append(label_path)
			continue
		pairs.append((image_path, label_path))

	return pairs, missing_images


def copy_pair(image_path: Path, label_path: Path, output_root: Path, split_name: str) -> None:
	dst_image = output_root / "images" / split_name / image_path.name
	dst_label = output_root / "labels" / split_name / label_path.name
	shutil.copy2(image_path, dst_image)
	shutil.copy2(label_path, dst_label)


def split_dataset(
	pairs: list[tuple[Path, Path]],
	val_ratio: float,
	seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
	if not 0 <= val_ratio <= 1:
		raise ValueError("val_ratio phải nằm trong khoảng [0, 1].")

	random.seed(seed)
	random.shuffle(pairs)

	val_count = int(len(pairs) * val_ratio)
	val_pairs = pairs[:val_count]
	train_pairs = pairs[val_count:]
	return train_pairs, val_pairs


def main() -> None:
	args = parse_args()

	image_dir = args.source_root / "image"
	label_dir = args.source_root / "label"

	if not image_dir.exists() or not label_dir.exists():
		raise FileNotFoundError(
			f"Không tìm thấy thư mục nguồn. Kiểm tra: {image_dir} và {label_dir}"
		)

	ensure_output_dirs(args.output_root)
	clear_split_dirs(args.output_root)

	pairs, missing_images = collect_pairs(image_dir, label_dir)
	if not pairs:
		raise RuntimeError("Không tìm thấy cặp image-label hợp lệ để chia dữ liệu.")

	train_pairs, val_pairs = split_dataset(pairs, args.val_ratio, args.seed)

	for image_path, label_path in train_pairs:
		copy_pair(image_path, label_path, args.output_root, "train")

	for image_path, label_path in val_pairs:
		copy_pair(image_path, label_path, args.output_root, "val")

	print("=== Hoan tat chia du lieu ===")
	print(f"Tong cap hop le: {len(pairs)}")
	print(f"Train: {len(train_pairs)}")
	print(f"Val: {len(val_pairs)}")
	if missing_images:
		print(f"Nhan khong tim thay anh tuong ung: {len(missing_images)}")


if __name__ == "__main__":
	main()
