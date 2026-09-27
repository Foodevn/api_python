import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


# =========================================================
# CONFIG
# =========================================================

JSON_DIR = Path(
    r"./Dataset/chua_co_label/labels_json"
)

OUTPUT_DIR = Path(
    r"./Dataset/chua_co_label/labels_txt"
)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.5


# =========================================================
# CONVERT 1 JSON
# =========================================================

def convert_json_to_txt(json_path):

    # -----------------------------------------------------
    # Đọc JSON
    # -----------------------------------------------------

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -----------------------------------------------------
    # JSON ngoài cùng là LIST
    # -----------------------------------------------------

    if not isinstance(data, list):
        print("  ! JSON không phải list")
        return 0

    txt_lines = []

    # =====================================================
    # Duyệt từng phần tử bên ngoài
    # =====================================================

    for item in data:

        if not isinstance(item, dict):
            continue

        # -------------------------------------------------
        # Lấy predictions
        # -------------------------------------------------

        prediction_container = item.get("predictions")

        if prediction_container is None:
            continue

        # Trường hợp:
        #
        # "predictions": {
        #     "predictions": [...]
        # }
        #
        if isinstance(prediction_container, dict):

            predictions = prediction_container.get(
                "predictions",
                []
            )

        # Trường hợp:
        #
        # "predictions": [...]
        #
        elif isinstance(prediction_container, list):

            predictions = prediction_container

        else:
            continue

        # -------------------------------------------------
        # Kiểm tra
        # -------------------------------------------------

        if not isinstance(predictions, list):
            continue

        # =================================================
        # DUYỆT TỪNG QUẢ DÂU
        # =================================================

        for prediction in predictions:

            if not isinstance(prediction, dict):
                continue

            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            confidence = prediction.get(
                "confidence",
                0
            )

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # -------------------------------------------------
            # Class
            # -------------------------------------------------

            class_id = prediction.get(
                "class_id",
                0
            )

            # -------------------------------------------------
            # RLE mask
            # -------------------------------------------------

            rle = prediction.get(
                "rle_mask"
            )

            if not isinstance(rle, dict):
                continue

            # =================================================
            # DECODE RLE
            # =================================================

            try:

                rle_decode = rle.copy()

                counts = rle_decode.get("counts")

                if isinstance(counts, str):
                    counts = counts.encode("utf-8")

                rle_decode["counts"] = counts

                mask = mask_utils.decode(
                    rle_decode
                )

            except Exception as e:

                print(
                    f"  ! Lỗi decode RLE: {e}"
                )

                continue

            # -------------------------------------------------
            # Mask về H x W
            # -------------------------------------------------

            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            mask = mask.astype(
                np.uint8
            )

            # =================================================
            # FIND CONTOUR
            # =================================================

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            # -------------------------------------------------
            # Lấy contour lớn nhất
            # -------------------------------------------------

            contour = max(
                contours,
                key=cv2.contourArea
            )

            # -------------------------------------------------
            # Bỏ object quá nhỏ
            # -------------------------------------------------

            area = cv2.contourArea(
                contour
            )

            if area < 10:
                continue

            # =================================================
            # SIMPLIFY POLYGON
            # =================================================

            perimeter = cv2.arcLength(
                contour,
                True
            )

            epsilon = 0.002 * perimeter

            contour = cv2.approxPolyDP(
                contour,
                epsilon,
                True
            )

            # -------------------------------------------------
            # Lấy các điểm polygon
            # -------------------------------------------------

            points = contour.reshape(
                -1,
                2
            )

            if len(points) < 3:
                continue

            # =================================================
            # KÍCH THƯỚC MASK
            # =================================================

            height, width = rle["size"]

            # =================================================
            # NORMALIZE
            # =================================================

            polygon = []

            for x, y in points:

                x_norm = x / width
                y_norm = y / height

                x_norm = max(
                    0.0,
                    min(1.0, x_norm)
                )

                y_norm = max(
                    0.0,
                    min(1.0, y_norm)
                )

                polygon.append(
                    x_norm
                )

                polygon.append(
                    y_norm
                )

            # =================================================
            # YOLO SEGMENTATION
            # =================================================

            line = (
                str(class_id)
                + " "
                + " ".join(
                    f"{value:.6f}"
                    for value in polygon
                )
            )

            txt_lines.append(line)

    # =====================================================
    # SAVE TXT
    # =====================================================

    output_path = (
        OUTPUT_DIR /
        f"{json_path.stem}.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(txt_lines)
        )

    return len(txt_lines)


# =========================================================
# MAIN
# =========================================================

json_files = list(
    JSON_DIR.glob("*.json")
)

print(
    f"Tìm thấy {len(json_files)} file JSON"
)

total_objects = 0

for index, json_path in enumerate(
    json_files,
    start=1
):

    try:

        count = convert_json_to_txt(
            json_path
        )

        total_objects += count

        print(
            f"[{index}/{len(json_files)}] "
            f"✓ {json_path.name} "
            f"→ {json_path.stem}.txt "
            f"({count} objects)"
        )

    except Exception as e:

        print(
            f"[{index}/{len(json_files)}] "
            f"✗ Lỗi {json_path.name}: {e}"
        )


print()
print("==============================")
print("Hoàn thành!")
print(
    f"Tổng JSON: {len(json_files)}"
)
print(
    f"Tổng objects: {total_objects}"
)
print(
    f"Labels: {OUTPUT_DIR}"
)
print("==============================")