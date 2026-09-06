import joblib

from src.final_evaluation.model import load_best_model_checkpoint, create_model
from src.final_evaluation.evaluate import (
    create_test_dataloader,
    generate_predictions,
    inverse_transform_targets,
    calculate_metrics,
    display_results,
    save_evaluation_report
    )
from src.final_evaluation.model_registry import log_to_mlflow
from src.utils.preprocessing import scaler_path



def main():

    print("=" * 60)
    print("BODYM FINAL MODEL EVALUATION")
    print("=" * 60)

    # Load model checkpoint

    model_checkpoint = load_best_model_checkpoint()

    batch_size = model_checkpoint["trial_params"]["batch_size"]

    # Reconstruct model

    model = create_model(model_checkpoint)

    # create test loader
    test_loader = create_test_dataloader(batch_size=batch_size)

    # load target scaler
    scaler = joblib.load(scaler_path)

    # Generate predictions
    scaled_predictions, scaled_targets = generate_predictions(
        model=model,
        test_loader=test_loader
    )

    # Convert to real measurement units
    predictions, targets = inverse_transform_targets(
        scaler,
        scaled_predictions=scaled_predictions,
        scaled_targets=scaled_targets
    )

    (
        overall_mae,
        overall_mse,
        overall_rmse,
        measurement_results
    ) = calculate_metrics(
        predictions=predictions,
        targets=targets
        )

    # Display result
    display_results(
        overall_mae=overall_mae,
        overall_mse=overall_mse,
        overall_rmse=overall_rmse,
        measurement_results=measurement_results
    )

    # Save report
    report_path = save_evaluation_report(measurement_results)

    # LOGGING TO MLflow
    run_id = log_to_mlflow(
        model=model,
        model_checkpoint=model_checkpoint,
        overall_mae=overall_mae,
        overall_mse=overall_mse,
        overall_rmse=overall_rmse,
        measurement_results=measurement_results,
        report_path=report_path
    )


    print("=" * 60)
    print("EVALUATION + MLFLOW COMPLETE")
    print("=" * 60)

    print(f"MLflow Run ID: {run_id}")

    print("The final model has been evaluated and logged to MLflow.")


if __name__ == "__main__":
    main()