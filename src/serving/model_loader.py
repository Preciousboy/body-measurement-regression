import mlflow
import mlflow.pytorch
import torch

REGISTERED_MODEL_NAME = "BodyM Regression Model"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# MODEL LOADER

def load_model():

    print("=" * 60)
    print("Loading production model from MLflow")
    print("=" * 60)

    print("MLflow Tracking URI: http://localhost:5000")
    print(F"Registered model: {REGISTERED_MODEL_NAME}")
    print("Model Alias: champion")
    print(f"Device: {DEVICE}")

    # configure MLflow
    mlflow.set_tracking_uri(
        "http://localhost:5000"
    )

    # Build the model URI

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

    print(f"Model URI: {model_uri}")

    # Load model from MLflow Model Registry
    model = mlflow.pytorch.load_model(
        model_uri=model_uri
    )

    # Move model to CPU/GPU
    model = model.to(DEVICE)

    model.eval()

    print("Production model loaded successfully....")

    return model