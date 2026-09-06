import mlflow

def log_to_mlflow(
        model,
        model_checkpoint,
        overall_mae,
        overall_mse,
        overall_rmse,
        measurement_results,
        report_path
    ):

    print("=" * 60)
    print("Logging Final evaluation to mlflow")
    print("=" * 60)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Final_model_evaluation")

    trial_params = model_checkpoint["trial_params"]

    with mlflow.start_run(
        run_name="final_model_evaluation"
    ):

        # Log parameters

        mlflow.log_params(
            {
                "trial_number": model_checkpoint["trial_number"],
                "dropout": trial_params["dropout"],
                "hidden_layers": trial_params["hidden_layers"],
                "learning_rate": trial_params["learning_rate"],
                "batch_size": trial_params["batch_size"],
                "optimizer": trial_params["optimizer"],
                "weight_decay": trial_params["weight_decay"],
                "epoch": model_checkpoint["epoch"],
                "best_val_loss": model_checkpoint["best_val_loss"]
            }
        )

        # Log overall test metrics

        mlflow.log_metrics(
            {
                "test_mae": overall_mae,
                "test_mse": overall_mse,
                "test_rmse": overall_rmse
            }
        )

        # Log per-measurement metrics

        for result in measurement_results:

            measurement = result["measurement"]
            
            mlflow.log_metrics(
                {
                    f"test_mae_{measurement}": result["mae_cm"],
                    f"test_mse_{measurement}": result["mse_cm2"],
                    f"test_rmse_{measurement}": result["rmse_cm"]
                }
            )

        # log evaluation report

        mlflow.log_artifact(
            str(report_path),
            artifact_path="evaluation"
        )

        # log and register model

       
        model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",
        registered_model_name="BodyM Regression Model",
        serialization_format="pickle"
        )

        print("Model logged and resgistered to Mlflow....")
        print(f"model URI: {model_info.model_uri}")
        print(f"Registered model: BodyM Regression Model")

        run_id = mlflow.active_run().info.run_id

        print(f"MLflow Run ID: {run_id}")

        return run_id
