import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="Fruit Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MODEL_PATH = Path(os.getenv("MODEL_PATH", "./runs/detect/train/weights/best.pt"))

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH.resolve()}")

# Load model once when server starts.
model = YOLO(str(MODEL_PATH))


@app.get("/health")
def health_check():
    return {"status": "ok", "model_path": str(MODEL_PATH)}

                                
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Use image/jpeg, image/png, or image/webp."
            ),
        )

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)

        file_size_mb = Path(temp_file_path).stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size is {MAX_UPLOAD_SIZE_MB}MB.",
            )

        results = await run_in_threadpool(
            model.predict,
            source=temp_file_path,
            save=False,
            verbose=False,
        )
        result = results[0]

        if len(result.boxes) == 0:
            return {"message": "No object detected", "detections": []}

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence": round(conf, 3),
                    "bbox_xyxy": [round(v, 2) for v in xyxy],
                }
            )

        best = max(detections, key=lambda item: item["confidence"])
        return {
            "best_prediction": {
                "class": best["class_name"],
                "confidence": best["confidence"],
            },
            "detections": detections,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        await file.close()
        if temp_file_path and Path(temp_file_path).exists():
            Path(temp_file_path).unlink()