from contextlib import asynccontextmanager

import torch

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.serving.model_loader import load_model, DEVICE
from src.serving.segmentation import HumanSegmenter, load_image_from_bytes
from src.serving.preprocessing import create_model_input
from src.serving.prediction import MeasurementPredictor
from src.serving.schema import PredictionResponse

# Global services

model = None
segmenter = None
predictor = None

# Application startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global segmenter
    global predictor

    print("=" * 60)
    print("FASTAPI STARTUP")
    print("=" * 60)

    # Load production regression model
    model = load_model()

    # Load human segmentation model
    segmenter = HumanSegmenter(DEVICE)

    # load target scaler and prediction service
    predictor = MeasurementPredictor(
        model=model,
        device=DEVICE
    )

    print("=" * 60)
    print("FASTAPI READY")
    print("=" * 60)

    yield

    print("=" * 60)
    print("FASTAPI SHUTDOWN")
    print("=" * 60)


# FastApi application

app = FastAPI(
    title="BodyM Measurement API",
    description=(
        "Body measurement prediction service using MLflow and FastAPI"
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Health endpoint

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "segmenter_loaded": segmenter is not None,
        "predictor_loaded": predictor is not None,
        "device": str(DEVICE)
    }

# Prediction endpoint

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...)
):

    # Validate content types

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if front_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=("front_image must be JPEG, PNG, or WEBP")
        )

    if side_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=("side_image must be JPEG, PNG or WEBP")
        )

    try:

        # READ UPLOADED FILES
        front_bytes = await front_image.read()
        side_bytes = await side_image.read()

        # Convert bytes to PIL RGB
        front_pil = load_image_from_bytes(front_bytes)
        side_pil = load_image_from_bytes(side_bytes)

        # segment both images
        front_mask = segmenter.segment(front_pil)
        side_mask = segmenter.segment(side_pil)

        # Reproduce BodyM preprocessing
        model_input = create_model_input(
            front_mask=front_mask,
            side_mask=side_mask
        )

        result = predictor.predict(
            model_input=model_input
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )