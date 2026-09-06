import mlflow
import optuna

from .objective import objective

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=5,
    interval_steps=1
)
sampler = optuna.samplers.TPESampler(
    seed=42
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(
    "Body measurement experiment"
)

study = optuna.create_study(
    study_name=(
        "Body measurement system"
    ),
    storage=(
        "sqlite:///optuna.db"
    ),
    load_if_exists=True,
    direction="minimize",
    pruner=pruner,
    sampler=sampler
)

study.optimize(
    objective,
    n_trials=20
)

print("Best hyperparameters")

for parameter, value in study.best_params.items():
    print(f"{parameter}: {value}")

