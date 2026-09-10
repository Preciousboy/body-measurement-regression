import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = (224, 224)

def preprocess_mask(
        mask: Image.Image
) -> np.ndarray:
    """
    Reproduce the exact mask preprocessing used
    by BodyMDataset.

    Training pipeline:

        grayscale
        ↓
        resize 224x224
        ↓
        float32
        ↓
        /255
    """
    
    # Match BodyMDataset:
    # Image.open(...).convert("L")

    mask = mask.convert("L")

    # Match BodyMDataset:
    # Image.Resampling.NEAREST 

    mask = mask.resize(
        IMAGE_SIZE,
        Image.Resampling.NEAREST
    )

    # Match BodyMDataset:
    # np.array(..., dtype=np.float32)
    mask = np.array(
        mask,
        dtype=np.float32
    )

    # Match BodyMDataset:
    # mask /= 255.0
    mask /= 255.0

    return mask

def create_model_input(
        front_mask: Image.Image,
        side_mask: Image.Image
) -> torch.Tensor:
    """
    Create the exact tensor representation expected
    by BodyMRegressionModel.

    Output shape:

        [1, 2, 224, 224]
    """

    front = preprocess_mask(
        mask=front_mask
    )

    side = preprocess_mask(
        mask=side_mask
    )

    # Equivalent to:
    #
    # np.stack(
    #     [mask, mask_left],
    #     axis=0
    # )
    #
    # from BodyMDataset.

    image = np.stack(
        [
            front,
            side
        ]
    )

    # Add batch dimension
    image = torch.tensor(
        image,
        dtype=torch.float32
    ).unsqueeze(0)

    return image