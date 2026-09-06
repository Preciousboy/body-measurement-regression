import torch

from src.final_evaluation.config import best_model_checkpoint_path
from src.models.model import BodyRegressionModel

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Loading the best modek checkpoint

def load_best_model_checkpoint():
    print("=" * 60)
    print("Loading best model checkpoint....")
    print("=" * 60)

    if not best_model_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best model checkpoint not found: {best_model_checkpoint_path}"
        )

    model_checkpoint = torch.load(
        best_model_checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    print(f"Trial number: {model_checkpoint["trial_number"]}")
    print(f"Training epoch: {model_checkpoint["epoch"]}")
    print(f"model val loss: {model_checkpoint["best_val_loss"]}")
    print("="*60)
    print("Model Hyperparameters")
    print("=" * 60)

    for parameter, value in model_checkpoint["trial_params"].items():
        print(f"{parameter}: {value}")

    return model_checkpoint

# Creating the model

def create_model(checkpoint):
    print("=" * 60)
    print("Reconstructing best model")
    print("=" * 60)

    trial_params = checkpoint["trial_params"]

    # Our model needs dropout and hidden_layers

    dropout = trial_params["dropout"]
    hidden_layers = trial_params["hidden_layers"]

    print(f"dropout: {dropout}")
    print(f"hidden_layers: {hidden_layers}")

    # Recreate our model

    model = BodyRegressionModel(
        dropout=dropout,
        hidden_layers=hidden_layers
    )

    # Load the best model learned parameters
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model reconstructed and weights loaded successfully")

    return model