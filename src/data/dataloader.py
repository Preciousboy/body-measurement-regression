import torch
from torch.utils.data import DataLoader

from .bodym_dataset import BodyMDataset
from .config import (
    Train_csv,
    Val_csv,
    Test_csv,
    Train_mask_dir,
    Train_mask_left_dir,
    Val_mask_dir,
    Val_mask_left_dir,
    Test_mask_dir,
    Test_mask_left_dir
)
from src.utils.preprocessing import preprocess_target
scaler, y_train_scaled, y_val_scaled, y_test_csaled = preprocess_target(
    Train_csv,
    Val_csv,
    Test_csv
)

def create_train_dataset():
    return BodyMDataset(
        csv_file=Train_csv,
        mask_dir=Train_mask_dir,
        mask_left_dir=Train_mask_left_dir,
        targets=y_train_scaled
    )

def create_val_dataset():
    return BodyMDataset(
        csv_file=Val_csv,
        mask_dir=Val_mask_dir,
        mask_left_dir=Val_mask_left_dir,
        targets=y_val_scaled
    )

def create_test_dataset():
    return BodyMDataset(
        csv_file=Test_csv,
        mask_dir=Test_mask_dir,
        mask_left_dir=Test_mask_left_dir,
        targets=y_test_csaled
    )

def dataloaders(batch_size):
    train_dataset = create_train_dataset()
    val_dataset = create_val_dataset()
    test_dataset = create_test_dataset

    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataloader = DataLoader(
            val_dataset,
            batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    test_dataloader = DataLoader(
            test_dataset,
            batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader
    )