from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset

# Target measurements
Target_columns = [
    "ankle",
    "arm-length",
    "bicep",
    "calf",
    "chest",
    "forearm",
    "height",
    "hip",
    "leg-length",
    "shoulder-breadth",
    "shoulder-to-crotch",
    "thigh",
    "waist",
    "wrist",
]

class BodyMDataset(Dataset):
    def __init__(self,
                 csv_file,
                 mask_dir,
                 mask_left_dir,
                 targets
                 ):
        """
        Parameters
        ----------
        csv_file:
            CSV containing subject_id and photo_id.

        mask_dir:
            Directory containing mask images.

        mask_left_dir:
            Directory containing mask_left images.

        targets:
            Preprocessed/scaled target array corresponding
            row-for-row with csv_file.

        image_size:
            Image size used by the model.
        """

        # Load metadata 
        self.df = pd.read_csv(csv_file)

        # Load mask and mask_left
        self.mask_dir = Path(mask_dir)
        self.mask_left_dir = Path(mask_left_dir)

        # convert traget to numpy array
        self.targets = np.asarray(
            targets,
            dtype = np.float32
        )
        self.size = (224, 224)

        # Safety check
        if len(self.df) != len(self.targets):
            raise ValueError(
                "Length of csv rows does not match length of target rows"
            )

        if self.targets.shape[1] != len(Target_columns):
            raise ValueError(
                f"Expected  {len(Target_columns)} but recieved {self.targets.shape[1]}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # Get metadata
        row = self.df.iloc[index]

        photo_id = str(row["photo_id"])
        subject_id = str(row["subject_id"])

        # Construct image paths
        mask_path = (
            self.mask_dir / f"{photo_id}.png"
        )
        mask_left_path = (
            self.mask_left_dir / f"{photo_id}.png"
        )

        # Check files
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        if not mask_left_path.exists():
            raise FileNotFoundError(
                f"Make_left not found: {mask_left_path}"
            )

        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask_left = Image.open(mask_left_path).convert("L")

        # Resize 
        mask = mask.resize(
            self.size,
            Image.Resampling.NEAREST
        )

        mask_left = mask_left.resize(
            self.size,
            Image.Resampling.NEAREST
        )

        # Convert image to numpy array
        mask = np.array(
            mask,
            dtype=np.float32
        )

        mask_left = np.array(
            mask_left,
            dtype=np.float32
        )

        # Perform normalization
        mask /= 255.0
        mask_left /= 255.0

        # Create two-channel image
        image = np.stack(
            [
                mask,
                mask_left
            ],
            axis=0
        )

        # Convert image to tensor
        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # Get our scaled target
        target = torch.tensor(
            self.targets[index],
            dtype=torch.float32
        )

        return (
            image,
            target,
            photo_id,
            subject_id
        )