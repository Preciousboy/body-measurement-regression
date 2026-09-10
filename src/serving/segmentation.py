from io import BytesIO

import numpy as np
import torch

from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights
)

class HumanSegmenter:
    """
    Converts an RGB human photograph into a binary
    human silhouette mask.

    Output:
        PIL.Image in mode "L"
        background = 0
        person = 255
    """

    def __init__(self, device: torch.device):

        self.device = device

        print("=" * 60)
        print("LOADING HUMAN SEGMENTATION MODEL")
        print("=" * 60)

        # loading pretrained weight
        self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights= self.weights)
        self.model = self.model.to(self.device)

        self.model.eval()

        # Torchvision provides the correct preprocessing for these pretrained weights

        self.preprocess = self.weights.transforms()

        # Find the "person" class dynamically
        categories = self.weights.meta["categories"]

        if "person" not in categories:
            raise RuntimeError("The segmentation model does not contain a person class")

        self.person_class_index = categories.index("person")

        print(f"Person class index: {self.person_class_index}")

        print("Human segmentation model loaded")

    def segment(self, image: Image.Image) -> Image.Image:
        """
        Segment the person from an RGB image.

        Parameters
        ----------
        image:
            PIL RGB image.

        Returns
        -------
        PIL.Image
            Binary grayscale mask.
        """

        # Make sure image is RGB
        image = image.convert("RGB")

        # Torchvision preprocessing
        input_tensor = self.preprocess(image)

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        # move to device
        input_tensor = input_tensor.to(self.device)

        # Run segmentation
        with torch.inference_mode():
            output = self.model(input_tensor)["out"]

        # Convert logits to probabilities
        probability = torch.softmax(output, dim= 1)

        person_probability = probability[
            0,
            self.person_class_index
        ]

        # Binary mask
        binary_mask = (
            person_probability >= 0.5
        )

        # Convert to uint8:
        #
        # False -> 0
        # True  -> 255

        mask = (
            binary_mask
            .cpu()
            .numpy()
            .astype(np.uint8)
            * 255
        )

        # convert Numpy array to PIL image
        mask_image = Image.fromarray(
            mask,
            mode="L"
        )

        
        # Resize back to original image dimensions.
        #
        # This is important because the segmentation model's
        # preprocessing may resize the image internally.

        mask_image = mask_image.resize(
            image.size,
            Image.Resampling.NEAREST
        )

        return mask_image


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Convert uploaded file bytes into a PIL RGB image.
    """

    image = Image.open(
        BytesIO(image_bytes)
        )

    return image.convert("RGB")