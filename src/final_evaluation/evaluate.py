import pandas as pd
import joblib
import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.bodym_dataset import BodyMDataset, Target_columns
from src.data.config import Root_dir, Test_csv, Test_mask_dir, Test_mask_left_dir
from src.utils.preprocessing import scaler_path

Target_column = [
    *Target_columns
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(Target_column)

test_df = pd.read_csv(Test_csv)

y_test = test_df[Target_column].values

scaler = joblib.load(scaler_path)

y_test_scaled = scaler.transform(y_test)

print(y_test_scaled)

# Create test dataloader

def create_test_dataloader(batch_size):
    print("=" * 60)
    print("Loading test dataset")
    print("=" * 60)

    test_dataset = BodyMDataset(
        Test_csv,
        Test_mask_dir,
        Test_mask_left_dir,
        y_test_scaled
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return test_dataloader

# Making predictions

def generate_predictions(
        model,
        test_loader
    ):

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, target, _, _ in test_loader:

            images = images.to(DEVICE)
            target = target.to(DEVICE)

            prediction = model(images)

            all_predictions.append(prediction.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    scaled_predictions = np.concatenate(
        all_predictions,
        axis=0
    )
    scaled_tragets = np.concatenate(
        all_targets,
        axis=0
    )

    print(f"scaled prediction shape: {scaled_predictions.shape}")
    print(f"scaled prediction target: {scaled_tragets.shape}")

    return scaled_predictions, scaled_tragets

# Inverse scaler transform

def inverse_transform_targets(
        scaler,
        scaled_predictions,
        scaled_targets
    ):
    print("=" * 60)
    print("Performing scaler transform to get the real measurement units")
    print("=" * 60)

    predictions = scaler.inverse_transform(scaled_predictions)
    targets = scaler.inverse_transform(scaled_targets)

    print("Prediction and targets converted back to original units")

    return predictions, targets

# Calculate metrics

def calculate_metrics(
        predictions,
        targets
    ):

    print("=" * 60)
    print("Calculating test metrics")
    print("=" * 60)

    overall_mae = mean_absolute_error(targets, predictions)
    overall_mse = mean_squared_error(targets, predictions)

    overall_rmse = np.sqrt(overall_mse)

    # Per measurement metrics

    measurement_results = []

    for index, measurement in enumerate(Target_column):
        actual = targets[:, index]
        predicted = predictions[:, index]

        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)

        measurement_results.append(
            {
                "measurement": measurement,
                "mae_cm": mae,
                "mse_cm2": mse,
                "rmse_cm": rmse
            }
        )

    return (
        overall_mae,
        overall_mse,
        overall_rmse,
        measurement_results
    )

# Display results

def display_results(
        overall_mae,
        overall_mse,
        overall_rmse,
        measurement_results
    ):

    print("=" * 60)
    print("Displaying final test results")
    print("=" * 60)

    print(f"overall MAE: {overall_mae:.4f} cm")
    print(f"overall mse: {overall_mse:.4f} cm2")
    print(f"overall rmse: {overall_rmse:.4f} cm")

    print("=" * 60)

    for result in measurement_results:
        print(
            f"{result['measurement']:<25}"
            f"{result['mae_cm']:>12.4f}"
            f"{result['rmse_cm']:>14.4f}"
            f"{result['mse_cm2']:>14.4f}"
            )


# Save CSV report

def save_evaluation_report(measurement_results):
    results_dir = Root_dir / f"artifacts/evaluation"

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = results_dir / f"test_measurement_metrics.csv"

    with open(report_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "measurement",
                "mae_cm",
                "mse_cm2",
                "rmse_cm"
            ]
        )

        writer.writeheader()

        writer.writerows(measurement_results)

    print(f"Evaluation result saved to: {report_path}")

    return report_path