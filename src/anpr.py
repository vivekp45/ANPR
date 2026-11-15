
import os

# Configure PaddleOCR before importing PaddlePaddle.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import re
import sys
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
from paddleocr import TextRecognition

# Import SQLite database lookup.
try:
    from database import lookup_vehicle
    DATABASE_AVAILABLE = True
    DATABASE_IMPORT_ERROR = None
except ImportError as error:
    lookup_vehicle = None
    DATABASE_AVAILABLE = False
    DATABASE_IMPORT_ERROR = error


# ---------------- CONFIGURATION ----------------

MODEL_PATH = "models/license-plate-finetune-v1l.pt"

OUTPUT_DIR = Path("output")
CROP_DIR = OUTPUT_DIR / "crops"

# Updated YOLO detection settings.
YOLO_CONFIDENCE = 0.15
YOLO_IMAGE_SIZE = 1600

OCR_MODEL_NAME = "en_PP-OCRv5_mobile_rec"

MIN_PLATE_WIDTH = 30
MIN_PLATE_HEIGHT = 12
MIN_OCR_CONFIDENCE = 0.50

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}


# ---------------- CHARACTER NORMALIZATION ----------------

_TO_DIGIT = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2", "S": "5", "B": "8",
    "G": "6", "A": "4"
})

_TO_LETTER = str.maketrans({
    "0": "O", "1": "I", "5": "S", "8": "B"
})


def clean_ocr_text(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def normalize_registration_text(raw_text):
    """
    Standard: State + RTO + 1-3 series letters + 4 digits.
    BH series: YYBH + 4 digits.

    Format validation does not guarantee OCR correctness.
    """

    text = clean_ocr_text(raw_text)

    if not text:
        return "", False

    # BH-series: YYBHNNNN
    if len(text) == 8:

        bh_candidate = (
            text[:2].translate(_TO_DIGIT)
            + "BH"
            + text[4:].translate(_TO_DIGIT)
        )

        if re.fullmatch(r"\d{2}BH\d{4}", bh_candidate):
            return bh_candidate, True

    # Standard Indian registration.
    for series_length in (3, 2, 1):

        expected_length = 4 + series_length + 4

        if len(text) != expected_length:
            continue

        state = text[:2].translate(_TO_LETTER)
        rto = text[2:4].translate(_TO_DIGIT)

        series = text[4:4 + series_length].translate(_TO_LETTER)
        serial = text[-4:].translate(_TO_DIGIT)

        candidate = state + rto + series + serial

        if (
            state in INDIAN_STATE_CODES
            and re.fullmatch(
                r"[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}",
                candidate
            )
        ):
            return candidate, True

    return text, False


# ---------------- OCR ----------------

def recognize_line(recognizer, image):

    if image is None or image.size == 0:
        return "", 0.0

    height, width = image.shape[:2]

    scale = max(2, min(6, 160 // max(height, 1)))

    enlarged = cv2.resize(
        image,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC
    )

    if len(enlarged.shape) == 3:
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    else:
        gray = enlarged.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(gray)

    thresholded = cv2.adaptiveThreshold(
        enhanced_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    variants = [
        enlarged,
        cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    ]

    best_text = ""
    best_score = 0.0

    for variant in variants:

        padded = cv2.copyMakeBorder(
            variant,
            15, 15, 20, 20,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        try:
            predictions = recognizer.predict(input=padded)

        except Exception as error:
            print(f"OCR variant failed: {error}")
            continue

        if not predictions:
            continue

        result = predictions[0]

        if isinstance(result, dict):
            text = result.get("rec_text", "")
            score = result.get("rec_score", 0.0)
        else:
            text = getattr(result, "rec_text", "")
            score = getattr(result, "rec_score", 0.0)

        if isinstance(text, (list, tuple)):
            text = "".join(str(item) for item in text)

        text = clean_ocr_text(text)

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        if text and score > best_score:
            best_text = text
            best_score = score

    return best_text, best_score


def get_ocr_text(recognizer, crop):

    if crop is None or crop.size == 0:
        return "", 0.0, False

    height, width = crop.shape[:2]
    candidates = []

    whole_text, whole_score = recognize_line(
        recognizer,
        crop
    )

    if whole_text:
        candidates.append((whole_text, whole_score))

    # Try separate recognition for two-line plates.
    aspect_ratio = width / max(height, 1)

    if aspect_ratio < 2.8 and height >= 24:

        midpoint = height // 2

        top_text, top_score = recognize_line(
            recognizer,
            crop[:midpoint, :]
        )

        bottom_text, bottom_score = recognize_line(
            recognizer,
            crop[midpoint:, :]
        )

        if top_text and bottom_text:

            combined_text = top_text + bottom_text
            combined_score = (top_score + bottom_score) / 2.0

            candidates.append(
                (combined_text, combined_score)
            )

        elif top_text or bottom_text:

            partial_text = top_text or bottom_text

            partial_score = (
                top_score if top_text else bottom_score
            )

            candidates.append(
                (partial_text, partial_score)
            )

    if not candidates:
        return "", 0.0, False

    evaluated = []

    for raw_text, score in candidates:

        normalized, format_valid = normalize_registration_text(
            raw_text
        )

        evaluated.append(
            (format_valid, score, normalized, raw_text)
        )

    best = max(
        evaluated,
        key=lambda item: (item[0], item[1])
    )

    format_valid, score, normalized, raw_text = best

    return normalized or raw_text, score, format_valid


# ---------------- DATABASE ----------------

def find_vehicle_record(registration_number):
    """
    Search SQLite only after registration format validation.
    """

    if not DATABASE_AVAILABLE or lookup_vehicle is None:
        return None, "DATABASE_UNAVAILABLE"

    try:
        record = lookup_vehicle(registration_number)

    except Exception as error:
        print(f"Database lookup failed: {error}")
        return None, "DATABASE_ERROR"

    if record:
        return record, "RECORD_FOUND"

    return None, "NO_RECORD_FOUND"


def print_vehicle_record(record):

    if not record:
        return

    print("\n  DATABASE RECORD")
    print("  -------------------------")

    record_type = str(
        record.get("record_type", "")
    ).upper()

    if "SYNTHETIC" in record_type or "FAKE" in record_type:

        print(
            "  WARNING: FICTIONAL / SYNTHETIC DEMO DATA"
            " — NOT REAL RTO DATA"
        )

    display_fields = [
        ("Registration", "registration_number_candidate"),
        ("Record type", "record_type"),
        ("Owner (fictional)", "owner_name_fake"),
        ("Vehicle class", "vehicle_class"),
        ("Manufacturer", "manufacturer_fake"),
        ("Model", "model_fake"),
        ("Variant", "variant_fake"),
        ("Colour", "colour_fake"),
        ("Fuel", "fuel_fake"),
        ("Manufacture year", "manufacture_year_fake"),
        ("State candidate", "state_from_prefix_candidate"),
        ("RTO code candidate", "rto_code_candidate"),
        ("Insurance status (synthetic)", "insurance_status_fake"),
        ("Insurance expiry (synthetic)", "insurance_expiry_fake"),
        ("PUC status (synthetic)", "puc_status_fake"),
        ("PUC expiry (synthetic)", "puc_expiry_fake"),
        ("Road tax status (synthetic)", "road_tax_status_fake"),
        ("Vehicle status (synthetic)", "vehicle_status_fake"),
        ("Source", "source"),
        ("Source image", "source_image"),
        ("Data quality", "data_quality_status")
    ]

    for label, key in display_fields:

        value = record.get(key)

        if value not in (None, ""):
            print(f"  {label}: {value}")


# ---------------- MODEL LOADING ----------------

def load_models():

    print("Loading YOLO license-plate detector...")

    detector = YOLO(MODEL_PATH)

    print("Loading PaddleOCR recognition model...")

    recognizer = TextRecognition(
        model_name=OCR_MODEL_NAME
    )

    print("Both models loaded successfully.\n")

    return detector, recognizer


# ---------------- FRAME PROCESSING ----------------

def process_frame(
    frame,
    detector,
    recognizer,
    crop_prefix="frame",
    save_crops=True
):

    annotated = frame.copy()
    plate_results = []

    # Updated detection:
    # Lower confidence and higher inference resolution.
    detections = detector.predict(
    source=frame,
    conf=YOLO_CONFIDENCE,
    imgsz=YOLO_IMAGE_SIZE,
    augment=True,
    verbose=False
)

    height, width = frame.shape[:2]
    plate_number = 0

    for detection in detections:

        if detection.boxes is None:
            continue

        for box in detection.boxes:

            coords = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coords.tolist()

            # Clamp coordinates.
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))

            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            if (
                box_width < MIN_PLATE_WIDTH
                or box_height < MIN_PLATE_HEIGHT
            ):
                continue

            plate_number += 1

            crop = frame[y1:y2, x1:x2].copy()

            crop_path = CROP_DIR / (
                f"{crop_prefix}_plate_{plate_number:02d}.jpg"
            )

            if save_crops:

                CROP_DIR.mkdir(
                    parents=True,
                    exist_ok=True
                )

                cv2.imwrite(str(crop_path), crop)

            # Compare original crop and trimmed crop.
            ocr_candidates = []

            crop_width = crop.shape[1]
            trim_pixels = int(crop_width * 0.085)

            candidate_crops = [crop]

            if 0 < trim_pixels < crop_width:
                candidate_crops.append(
                    crop[:, trim_pixels:]
                )

            for candidate_crop in candidate_crops:

                if candidate_crop.size == 0:
                    continue

                (
                    candidate_text,
                    candidate_score,
                    candidate_valid
                ) = get_ocr_text(
                    recognizer,
                    candidate_crop
                )

                if candidate_text:

                    ocr_candidates.append(
                        (
                            candidate_valid,
                            candidate_score,
                            candidate_text
                        )
                    )

            if ocr_candidates:

                format_valid, ocr_confidence, text = max(
                    ocr_candidates,
                    key=lambda item: (item[0], item[1])
                )

            else:
                text = ""
                ocr_confidence = 0.0
                format_valid = False

            yolo_confidence = float(
                box.conf[0].item()
            )

            if (
                not text
                or ocr_confidence < MIN_OCR_CONFIDENCE
            ):
                continue

            status = (
                "FORMAT_OK"
                if format_valid
                else "REVIEW_FORMAT"
            )

            # Database lookup.
            if format_valid:

                database_record, database_status = (
                    find_vehicle_record(text)
                )

            else:
                database_record = None
                database_status = "SKIPPED_REVIEW_FORMAT"

            plate_results.append({
                "plate": text,
                "yolo_confidence": yolo_confidence,
                "ocr_confidence": ocr_confidence,
                "format_valid": format_valid,
                "status": status,
                "database_status": database_status,
                "database_record": database_record,
                "box": (x1, y1, x2, y2),
                "crop_path": str(crop_path)
            })

            # Annotate plate.
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{text} ({ocr_confidence:.2f})"

            if not format_valid:
                label += " REVIEW"

            elif database_status == "RECORD_FOUND":
                label += " DB"

            elif database_status == "NO_RECORD_FOUND":
                label += " NO-DB"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            (
                (text_width, text_height),
                baseline
            ) = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            label_y = max(y1, text_height + 10)

            label_x2 = min(
                width - 1,
                x1 + text_width + 8
            )

            cv2.rectangle(
                annotated,
                (
                    x1,
                    label_y - text_height - 8
                ),
                (
                    label_x2,
                    label_y + baseline
                ),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y - 4),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

    return annotated, plate_results


# ---------------- IMAGE PROCESSING ----------------

def process_image(image_path, detector, recognizer):

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing image: {image_path}")

    annotated, results = process_frame(
        image,
        detector,
        recognizer,
        crop_prefix=image_path.stem,
        save_crops=True
    )

    output_path = OUTPUT_DIR / f"{image_path.stem}_anpr.jpg"

    if not cv2.imwrite(str(output_path), annotated):
        print(f"ERROR: Could not save {output_path}")
        return

    print("\n----------- ANPR RESULTS -----------")

    if not results:

        print(
            "No accepted license plates detected "
            "(check detection, size and OCR confidence filters)."
        )

    for index, result in enumerate(results, start=1):

        print(f"\nPlate {index}: {result['plate']}")

        print(
            f"YOLO confidence: "
            f"{result['yolo_confidence']:.3f}"
        )

        print(
            f"OCR confidence: "
            f"{result['ocr_confidence']:.3f}"
        )

        print(f"Format check: {result['status']}")

        print(
            f"Database status: "
            f"{result['database_status']}"
        )

        if result["database_record"]:

            print_vehicle_record(
                result["database_record"]
            )

        elif result["database_status"] == "NO_RECORD_FOUND":

            print(
                "  No matching record in local demo database."
            )

        elif result["database_status"] == "SKIPPED_REVIEW_FORMAT":

            print(
                "  Database lookup skipped: "
                "registration needs manual review."
            )

        elif result["database_status"] in (
            "DATABASE_UNAVAILABLE",
            "DATABASE_ERROR"
        ):

            print(
                "  Database lookup could not be completed."
            )

        print(f"Bounding box: {result['box']}")
        print(f"Crop saved: {result['crop_path']}")

    print(f"\nAnnotated image: {output_path}")
    print(f"Individual crops: {CROP_DIR}")


# ---------------- VIDEO PROCESSING ----------------

def process_video(video_path, detector, recognizer):

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_path = OUTPUT_DIR / f"{video_path.stem}_anpr.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():

        print("ERROR: Could not create output video.")

        cap.release()
        return

    sightings = Counter()
    frame_number = 0

    print(f"\nProcessing video: {video_path}")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            annotated, results = process_frame(
                frame,
                detector,
                recognizer,
                crop_prefix=(
                    f"{video_path.stem}_frame_"
                    f"{frame_number:06d}"
                ),
                save_crops=False
            )

            for result in results:

                if (
                    result["plate"]
                    and result["format_valid"]
                ):
                    sightings[result["plate"]] += 1

            writer.write(annotated)

            if frame_number % 30 == 0:
                print(
                    f"Processed {frame_number} frames..."
                )

    finally:
        cap.release()
        writer.release()

    print(f"\nAnnotated video: {output_path}")

    print("\nPlate sightings (per-frame detections):")

    for plate, count in sightings.most_common():
        print(f"{plate}: {count}")

    print(
        "\nNote: these are frame-level sightings, "
        "not unique vehicle counts."
    )


# ---------------- MAIN ----------------

def main():

    if len(sys.argv) != 2:

        print(
            'Usage: python anpr.py "sample inputs/0001.jpg"'
        )

        print(
            '   or: python anpr.py "videos/traffic.mp4"'
        )

        return

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    if not Path(MODEL_PATH).exists():
        print(f"ERROR: YOLO model not found: {MODEL_PATH}")
        return

    if DATABASE_AVAILABLE:
        print("SQLite vehicle database module loaded.")

    else:
        print("WARNING: Database module is unavailable.")

        print(
            f"Database import detail: {DATABASE_IMPORT_ERROR}"
        )

    detector, recognizer = load_models()

    extension = input_path.suffix.lower()

    if extension in IMAGE_EXTENSIONS:

        process_image(
            input_path,
            detector,
            recognizer
        )

    elif extension in VIDEO_EXTENSIONS:

        process_video(
            input_path,
            detector,
            recognizer
        )

    else:

        print(f"Unsupported file type: {extension}")


if __name__ == "__main__":
    main()
