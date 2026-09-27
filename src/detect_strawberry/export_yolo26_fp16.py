from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "../../runs/runs_backup_v7/runs/strawberry_seg/weights/best.pt"
IMGSZ = 1024


# ============================================================
# MAIN
# ============================================================

def main():

    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path}"
        )

    print("=" * 60)
    print("YOLO26s-seg -> OpenVINO FP16")
    print("=" * 60)

    print(f"Model: {model_path}")
    print(f"Image size: {IMGSZ}")
    print()

    # Load YOLO model
    model = YOLO(str(model_path))

    # Export OpenVINO FP16
    exported_path = model.export(
        format="openvino",
        imgsz=IMGSZ,
        half=True,
        dynamic=False,
    )

    print()
    print("=" * 60)
    print("EXPORT COMPLETED")
    print("=" * 60)

    print(f"Export path: {exported_path}")

    # Ultralytics thường tạo thư mục:
    #
    # best_openvino_model/
    # ├── best.xml
    # ├── best.bin
    # └── metadata.yaml

    exported_path = Path(exported_path)

    if exported_path.exists():

        print("\nGenerated files:")

        for file in exported_path.rglob("*"):

            if file.is_file():

                size_mb = file.stat().st_size / (1024 * 1024)

                print(
                    f"  {file.name:<30} "
                    f"{size_mb:.2f} MB"
                )

    print()
    print("Có thể dùng model XML để OpenVINO inference.")


if __name__ == "__main__":
    main()