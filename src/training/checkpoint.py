import torch

# Checkpoint path
from src.data.config import Root_dir

checkpoint_dir = Root_dir/f"checkpoints"
last_state_path = checkpoint_dir / f"last_state.pt"

# Save last state

def save_last_state(
        trial_number,
        epoch,
        model,
        optimizer,
        scheduler,
        trial_best_val_loss,
        train_loss,
        val_loss,
        trial_params
    ):

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Build checkpoints

    checkpoint = {
        "trial_number": trial_number,
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "trial_best_val_loss": trial_best_val_loss,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "trial_params": trial_params
    }

    # Save checkpoint

    torch.save(
        checkpoint,
        last_state_path
    )

def load_last_state():

    if not last_state_path.exists():
        return None

    checkpoint = torch.load(
        last_state_path,
        map_location="cpu"
    )
    return checkpoint

# Check[oint exists

def checkpoint_exists():
    """
    Check whether a recovery checkpoint exists
    """
    return last_state_path.exists()

def delete_last_state():
    """
    Delete the recovery checkpoint.

    This happens after the associated trial
    successfully finishes.
    """

    if last_state_path.exists():
        last_state_path.unlink()