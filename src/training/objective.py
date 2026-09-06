import os
import torch
import torch.nn as nn
import mlflow
import optuna

from src.training.training import training_loop
from src.training.evaluate import validating_loop
from src.models.model import BodyRegressionModel
from src.data.dataloader import dataloaders
from src.training.checkpoint import (
    save_last_state,
    load_last_state,
    delete_last_state
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

best_model_dir = "checkpoints/best_models"
os.makedirs(
    best_model_dir,
    exist_ok=True
)

epochs = 40
best_val_loss = float("inf")

def objective(trial):

    global best_val_loss

    trial_best_val_loss = float("inf")

    dropout = trial.suggest_float(
        "dropout",
        0.2,
        0.5
    )

    hidden_layers = trial.suggest_categorical(
        "hidden_layers",
        [32, 64, 128, 256]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        5e-6,
        1e-3,
        log=True
    )


    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64]
    )

    model = BodyRegressionModel(
    dropout= dropout,
    hidden_layers= hidden_layers
    )

    model = model.to(DEVICE)

    optimizer_name = trial.suggest_categorical(
    "optimizer",
    ["SGD", "Adam", "AdamW"]
    )

    if optimizer_name == "SGD":

        momentum = trial.suggest_float(
            "momentum",
            0.8,
            0.99
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )

    elif optimizer_name == "Adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    elif optimizer_name == "AdamW":

        weight_decay = trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-2,
            log=True
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    criterion = nn.MSELoss()

    train_dataloader, val_dataloader, test_dataloader = dataloaders(
        batch_size= batch_size
    )


    #-------------- CHECKPOINT LOGIC ---------------
    start_epoch = 0
    

    checkpoint = load_last_state()

    if checkpoint is not None:
        checkpoint_trial  = (
            checkpoint["trial_number"]
        )
        checkpoint_params = (
            checkpoint["trial_params"]
        )


        # making sure the checkpoint belongs to this trial
        if (
            checkpoint_trial == trial.number
            and checkpoint_params == trial.params
        ):
            print(
                "Recovery checkpoint found."
            )

            print(
                f"Trial: {checkpoint_trial}"
            )

            print(
                f"Last completed epoch: "
                f"{checkpoint['epoch'] + 1}"
            )

            # Restore model 

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            # Restore optimizer
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

            # Restore scheduler
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

            # Restore training state
            trial_best_val_loss = (
                checkpoint[
                    "trial_best_val_loss"
                ]
            )

            start_epoch = (
                checkpoint["epoch"] + 1
            )

        else:

            print()
            print(
                "Checkpoint exists, but it "
                "does not belong to this trial."
            )

            print(
                "Starting this trial from "
                "epoch 1."
            )

    with mlflow.start_run(
        run_name=f"optuna_trial_{trial.number}"
    ):

        # log hyperparameters

        mlflow.log_params(
            {   
                "trial_number": trial.number,
                "dropout": dropout,
                "hidden_layers": hidden_layers,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epoch": epochs,
                "optimizer": optimizer_name
            }
        )

        # Log optimizer specific parameter

        if optimizer_name == "SGD":

            mlflow.log_param(
                "momentum",
                momentum
            )

        elif optimizer_name == "AdamW":

            mlflow.log_param(
                "weight_decay",
                weight_decay
            )

        for epoch in range(start_epoch, epochs):

            # -------- Training loop----------


            training_loss = training_loop(
                                    model= model,
                                    optimizer= optimizer,
                                    loss_fn=criterion,
                                    dataloader=train_dataloader,
                                    device=DEVICE
                                )

            # ------- Validation loop -----------

            val_loss = validating_loop(
                                model=model,
                                dataloader=val_dataloader,
                                loss_fn=criterion,
                                device=DEVICE
                            )

            #----------------Scheduler------------
            scheduler.step(val_loss)

            # Update best validation loss

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                # Saving the state of the best_val_loss

                torch.save(
                        {
                            "trial_number": trial.number,
                            "trial_params": trial.params,
                            "epoch": epoch,
                            "best_val_loss": best_val_loss,

                            "model_state_dict": model.state_dict(),

                            "optimizer_state_dict": optimizer.state_dict(),

                            "scheduler_state_dict": scheduler.state_dict()
                        },
                        os.path.join(
                            best_model_dir,
                            "best_model.pt"
                        )
                    )
                print("=" * 60)
                print(f"Best model saved with loss of {best_val_loss} at trial {trial.number} in epoch {epoch}")
                print("=" * 60)


            if val_loss < trial_best_val_loss:
                trial_best_val_loss = val_loss

            # mlflow metrics
            mlflow.log_metrics(
                {
                "training_loss": training_loss,
                "val_loss": val_loss,
                "trial_best_val_loss": trial_best_val_loss,
                "global_best_val_loss": best_val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"]
            },
            step=epoch
            )

            # optuna reporting
            trial.report(
                val_loss,
                step=epoch
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {training_loss:.4f} "
                f"Val Loss: {val_loss:.4f}"
            )

            # ---------- SAVE LAST STATE---------------

            save_last_state(
                trial_number=trial.number,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                trial_best_val_loss=trial_best_val_loss,
                train_loss=training_loss,
                val_loss=val_loss,
                trial_params=trial.params
            )

            # Optuna pruning

            if trial.should_prune():
                print(
                    f"Trial {trial.number} "
                    f"was pruned."
                )

                # Remove recovery checkpoint

                delete_last_state()
                raise optuna.TrialPruned     

        print(
        f"Trial {trial.number} "
        f"completed successfully."
        )

        print(
            f"Final validation loss: "
            f"{val_loss:.6f}"
        )

        # Delete temporary recovery checkpoint

        delete_last_state()    

    return trial_best_val_loss