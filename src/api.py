
import os
import sys
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.chdir(PROJECT_ROOT)

SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "output"
CROP_DIR = OUTPUT_DIR / "crops"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROP_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))


# Import the existing ANPR pipeline.
from anpr import load_models, process_frame


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\nStarting ANPR API...")

    detector, recognizer = load_models()

    app.state.detector = detector
    app.state.recognizer = recognizer

    print("ANPR models are ready.")

    yield

    print("ANPR API shutting down.")


app = FastAPI(
    title="ANPR Image Recognition API",
    description="YOLO and PaddleOCR based license plate recognition",
    version="1.0.0",
    lifespan=lifespan
)


# Serve processed images and plate crops.
app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output"
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health_check():

    return {
        "status": "online",
        "application": "ANPR Image Recognition",
        "models": "YOLO + PaddleOCR"
    }


# --------------------------------------------------
# IMAGE UPLOAD AND ANPR PROCESSING
# --------------------------------------------------

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):

    allowed_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    request_id = uuid.uuid4().hex[:12]

    try:

        annotated, results = process_frame(
            image,
            app.state.detector,
            app.state.recognizer,
            crop_prefix=request_id,
            save_crops=True
        )

        output_filename = f"{request_id}_anpr.jpg"

        output_path = OUTPUT_DIR / output_filename

        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(
                "Could not save the processed image."
            )

        formatted_results = []

        for result in results:

            crop_path = Path(result["crop_path"])

            record = result.get("database_record")

            record_type = (
                str(record.get("record_type", "")).upper()
                if record else ""
            )

            formatted_results.append({

                "plate": result["plate"],

                "yolo_confidence": round(
                    float(result["yolo_confidence"]), 4
                ),

                "ocr_confidence": round(
                    float(result["ocr_confidence"]), 4
                ),

                "format_valid": bool(
                    result["format_valid"]
                ),

                "status": result["status"],

                "database_status": result["database_status"],

                "database_record": record,

                "synthetic_data": (
                    "SYNTHETIC" in record_type
                    or "FAKE" in record_type
                ),

                "bounding_box": list(result["box"]),

                "crop_url": (
                    f"/output/crops/{crop_path.name}"
                )
            })

        return {

            "success": True,

            "filename": file.filename,

            "image_dimensions": {
                "width": int(image.shape[1]),
                "height": int(image.shape[0])
            },

            "total_plates_detected": len(formatted_results),

            "results": formatted_results,

            "processed_image_url": (
                f"/output/{output_filename}"
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        print(f"ANPR processing error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Image processing failed. Check the server terminal."
        )
# --------------------------------------------------
# FRONTEND WEBPAGE
# --------------------------------------------------

FRONTEND_DIR = PROJECT_ROOT / "app"

app.mount(
    "/",
    StaticFiles(
        directory=str(FRONTEND_DIR),
        html=True
    ),
    name="frontend"
)