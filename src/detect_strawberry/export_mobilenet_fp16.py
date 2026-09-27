from pathlib import Path

import torch
import torch.nn as nn
import openvino as ov

from torchvision.models import mobilenet_v3_small


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = "../../runs/mobilenetv3_small/best_v3.pth"

OUTPUT_DIR = Path(
    "../../runs/mobilenetv3_small/mobilenetv3_small_fp16"
)

NUM_CLASSES = 2

IMAGE_SIZE = 224


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():
    """
    Tạo kiến trúc MobileNetV3-Small giống lúc training.
    """

    model = mobilenet_v3_small(
        weights=None
    )

    # MobileNetV3-Small mặc định:
    #
    # classifier:
    #   Linear(576 -> 1024)
    #   Hardswish
    #   Dropout
    #   Linear(1024 -> 1000)
    #
    # Dataset:
    #   0 = xau
    #   1 = dep

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model):
    """
    Load checkpoint .pth vào MobileNetV3-Small.
    """

    print(
        f"Loading checkpoint: {CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Xác định state_dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            # Trường hợp:
            # torch.save(model.state_dict(), path)

            state_dict = checkpoint

    else:

        raise RuntimeError(
            "File .pth không phải state_dict."
        )

    # --------------------------------------------------------
    # Xử lý DataParallel
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# EXPORT OPENVINO FP16
# ============================================================

def export_openvino(model):

    model.eval()

    # --------------------------------------------------------
    # Example input
    # --------------------------------------------------------

    print()
    print("Creating example input...")

    example_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # PyTorch -> OpenVINO
    # --------------------------------------------------------

    print(
        "Converting PyTorch -> OpenVINO..."
    )

    ov_model = ov.convert_model(
        model,
        example_input=example_input
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        "mobilenetv3_small_fp16.xml"
    )

    # --------------------------------------------------------
    # Save OpenVINO model
    #
    # OpenVINO 2026.4:
    #
    # compress_to_fp16=True
    # sẽ nén weights sang FP16
    # --------------------------------------------------------

    print()
    print(
        "Saving OpenVINO model as FP16..."
    )

    ov.save_model(
        ov_model,
        str(output_path),
        compress_to_fp16=True
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EXPORT COMPLETED")
    print("=" * 60)

    print()

    print(
        f"XML: {output_path}"
    )

    bin_path = output_path.with_suffix(
        ".bin"
    )

    if output_path.exists():

        size_mb = (
            output_path.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"{output_path.name}: "
            f"{size_mb:.2f} MB"
        )

    if bin_path.exists():

        size_mb = (
            bin_path.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"{bin_path.name}: "
            f"{size_mb:.2f} MB"
        )

    print()

    print("Model files:")

    print(
        f"  {output_path}"
    )

    print(
        f"  {bin_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "MobileNetV3-Small -> OpenVINO FP16"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    checkpoint = Path(
        CHECKPOINT_PATH
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Không tìm thấy checkpoint:\n"
            f"{checkpoint.resolve()}"
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    print(
        f"Input: {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    model = load_checkpoint(
        model
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    export_openvino(
        model
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()