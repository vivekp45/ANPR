# ANPR - Automated Number Plate Recognition

An image-based license plate recognition application using YOLO, PaddleOCR, OpenCV, FastAPI and SQLite.

## Features
- License plate detection using YOLO.
- Character recognition using PaddleOCR.
- Indian registration number normalization and validation.
- SQLite vehicle database lookup.
- Image upload through a web interface.
- Annotated images and cropped number plates.
- YOLO and OCR confidence scores.

## Technology Stack
- Python 3.13
- YOLO (Ultralytics)
- PaddleOCR
- OpenCV
- FastAPI and Uvicorn
- SQLite
- HTML, CSS and JavaScript

## Installation

Clone the repository:

    git clone https://github.com/vivekp45/ANPR.git
    cd ANPR

Install dependencies:

    python -m pip install -r requirements.txt

## Run the Application

From the project root:

    python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000

Open the application:

    http://127.0.0.1:8000/

API documentation:

    http://127.0.0.1:8000/docs

## Processing Workflow
1. Upload a vehicle image.
2. YOLO detects the number plate.
3. OpenCV extracts the plate region.
4. PaddleOCR recognizes the registration characters.
5. The system normalizes and validates the registration number.
6. SQLite searches for a matching vehicle record.
7. Results and annotated images are displayed.

## Database Disclaimer

The included SQLite database contains fictional/synthetic demonstration records. It is not an official RTO database and must not be used for actual vehicle verification.

The system can recognize plates that are not present in the database. Such plates are reported as NO_RECORD_FOUND.

## Limitations
- Currently supports image-based recognition.
- Recognition depends on image quality, lighting and plate visibility.
- OCR results may require manual review.
- Database lookup is limited to locally available records.
- Running the application requires a compatible environment with sufficient resources for the deep learning models.

## Purpose

Developed for academic and demonstration purposes.
