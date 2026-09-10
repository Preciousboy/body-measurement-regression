import joblib
import numpy as np
import torch

from src.data.config import Root_dir
from src.data.bodym_dataset import Target_columns

SCALAR_PATH = Root_dir / f"checkpoints/preprocessed_target.pkl"
TARGET_COLUMNS = [
    *Target_columns
]

class MeasurementPredictor:

    def __init__(
            self,
            model,
            device,
            scaler_path = SCALAR_PATH
    ):

        self.model = model
        self.device = device

        # Load the scaler that was fitted on training targets
        self.scaler = joblib.load(
            scaler_path
        )

        print(
            "Target scaler loaded"
        )

    def predict(
            self,
            model_input: torch.Tensor
    ):

        # Move input to same device as model

        model_input = model_input.to(self.device)

        # Inference only

        with torch.inference_mode():
            scaled_prediction = self.model(model_input)

        # Move prediction to CPU
        scaled_prediction = (
            scaled_prediction
            .cpu()
            .numpy()
        )

        # Convert standardized predictions back to real measurement units
        prediction = (
            self.scaler.inverse_transform(
                scaled_prediction
            )
        )

        # Remove batch dimension
        prediction = prediction[0]

        # Build response dictionary
        result = {}

        for index, column in enumerate(TARGET_COLUMNS):
            result[column] = float(
                prediction[index]
            )

        return result