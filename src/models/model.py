import torch
import torch.nn as nn

class BodyRegressionModel(nn.Module):
    """
    CNN model for the prediction of the body measurement of an individual using
    there front and side picture
    """

    def __init__(self,
                 dropout,
                 hidden_layers
                 ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=2,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3

            nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                2,
                224,
                224
            )

            feature_output = self.features(dummy)

            flattened_features = feature_output.flatten(1).shape[1]

        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flattened_features, 
                      hidden_layers),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_layers, 14)
        )

    def forward(self, x):
        return self.regressor(self.features(x))

    
# model = BodyRegressionModel(0.2, 64)

# image = torch.randn(1, 2, 224, 224)

# output = model(image)
# print(f"image shape: {image.shape}")
# print(f"output shape: {output.shape}")