# Body Measurement Regression

A PyTorch-based computer vision regression system for predicting human body measurements from paired front-view and side-view images.

The project implements an end-to-end machine learning workflow covering data preprocessing, CNN model development, hyperparameter optimization with Optuna, checkpoint-based training recovery, experiment tracking with MLflow, final test-set evaluation, and model registration.

## Project Overview

The objective of this project is to predict **14 body measurements** from two complementary images of an individual:

- Front-view image
- Side-view image

The two images are represented as a two-channel input tensor and passed through a convolutional neural network designed for multi-output regression.

The overall workflow is:

```text
Front + Side Images
        │
        ▼
Data preprocessing
        │
        ▼
PyTorch DataLoader
        │
        ▼
CNN regression model
        │
        ▼
Optuna hyperparameter optimization
        │
        ▼
Best validation model
        │
        ▼
Final test-set evaluation
        │
        ▼
MLflow experiment tracking
        │
        ▼
MLflow Model Registry
```

## Model Architecture

The model is implemented in:

```text
src/models/model.py
```

The `BodyRegressionModel` is a convolutional neural network consisting of four convolutional blocks.

### Convolutional feature extractor

The network uses:

1. `Conv2d(2 → 32)`
2. `Conv2d(32 → 64)`
3. `Conv2d(64 → 128)`
4. `Conv2d(128 → 256)`

The convolutional layers use:

- `3 × 3` kernels
- padding of `1`
- Batch Normalization
- ReLU activation

The first three blocks also use max pooling to progressively reduce spatial dimensions.

The final convolutional block uses:

```python
AdaptiveAvgPool2d((1, 1))
```

This produces a fixed-size feature representation regardless of the spatial dimensions entering the adaptive pooling layer.

### Regression head

The extracted features are passed through a fully connected regression head containing:

- configurable hidden-layer size
- ReLU activation
- Dropout
- second configurable hidden layer
- ReLU activation
- Dropout
- final linear layer

The final layer produces:

```text
14 outputs
```

corresponding to the 14 predicted body measurements.

The following hyperparameters are configurable:

- `dropout`
- `hidden_layers`

## Hyperparameter Optimization

Hyperparameter optimization is implemented with **Optuna**.

The optimization objective is defined in:

```text
src/training/objective.py
```

Optuna searches over the following parameters:

| Hyperparameter     | Search Space              |
| ------------------ | ------------------------- |
| Dropout            | `0.2 – 0.5`               |
| Hidden layers      | `32, 64, 128, 256`        |
| Learning rate      | `5e-6 – 1e-3` (log scale) |
| Batch size         | `32, 64`                  |
| Optimizer          | `SGD, Adam, AdamW`        |
| SGD momentum       | `0.8 – 0.99`              |
| AdamW weight decay | `1e-6 – 1e-2` (log scale) |

The optimization objective is to minimize validation loss.

### Optimizers

The project evaluates three optimizers:

- SGD
- Adam
- AdamW

Optimizer-specific hyperparameters are only sampled when relevant.

For example, momentum is sampled for SGD, while weight decay is sampled for AdamW.

### Learning-rate scheduling

The training process uses:

```python
torch.optim.lr_scheduler.ReduceLROnPlateau
```

The scheduler reduces the learning rate when validation loss stops improving.

Configuration:

```text
mode = min
factor = 0.5
patience = 3
```

### Loss function

Training uses:

```python
nn.MSELoss()
```

because the task is a multi-output continuous regression problem.

## Checkpointing

The project implements checkpoint-based training recovery.

Checkpoint functionality is located in:

```text
src/training/checkpoint.py
```

The recovery system stores information including:

- trial number
- trial parameters
- current epoch
- model state
- optimizer state
- scheduler state
- best validation loss
- training loss
- validation loss

This allows an interrupted Optuna trial to resume from its saved state when the checkpoint belongs to the same trial and parameter configuration.

Temporary recovery checkpoints are removed after a trial completes or is pruned.

## MLflow Experiment Tracking

MLflow is used to track the hyperparameter optimization experiments and final model evaluation.

The training process records information such as:

- trial number
- dropout
- hidden-layer size
- learning rate
- batch size
- optimizer
- optimizer-specific parameters
- training loss
- validation loss
- trial best validation loss
- global best validation loss
- learning rate during training

The final evaluation pipeline also logs the evaluated model and its test metrics to MLflow.

The project uses the MLflow Model Registry to register the resulting model.

## Final Evaluation

The final evaluation pipeline is located in:

```text
src/final_evaluation/
```

The main entry point is:

```text
src/final_evaluation/main.py
```

The pipeline:

1. Loads the best model checkpoint.
2. Retrieves the hyperparameters used by the best trial.
3. Reconstructs the model architecture.
4. Creates the test DataLoader.
5. Loads the target scaler.
6. Generates predictions.
7. Converts predictions back to the original measurement units.
8. Calculates overall regression metrics.
9. Calculates per-measurement results.
10. Saves the evaluation report.
11. Logs the final model and evaluation results to MLflow.

The evaluation metrics include:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Project Structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt
│
└── src/
    ├── __init__.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── bodym_dataset.py
    │   ├── config.py
    │   └── dataloader.py
    │
    ├── models/
    │   ├── __init__.py
    │   └── model.py
    │
    ├── training/
    │   ├── __init__.py
    │   ├── checkpoint.py
    │   ├── evaluate.py
    │   ├── objective.py
    │   ├── study.py
    │   └── training.py
    │
    ├── final_evaluation/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── evaluate.py
    │   ├── main.py
    │   ├── model.py
    │   └── model_registry.py
    │
    └── utils/
        ├── __init__.py
        └── preprocessing.py
```

## Directory Responsibilities

### `src/data/`

Contains dataset and DataLoader functionality.

### `src/models/`

Contains the primary neural network architecture.

### `src/training/`

Contains:

- training loops
- validation
- Optuna objective
- study configuration
- checkpoint management

### `src/final_evaluation/`

Contains the final model reconstruction, test-set evaluation, reporting, and MLflow model registration workflow.

### `src/utils/`

Contains shared utility functionality such as preprocessing configuration.

## Environment

The project was developed using:

```text
Python 3.14.3
PyTorch 2.12.1
MLflow 3.15.1
Optuna 4.9.0
NumPy 2.4.3
Pandas 2.3.3
Scikit-learn 1.8.0
Joblib 1.5.3
tqdm 4.67.3
```

The complete dependency specification is provided in:

```text
requirements.txt
```

## Installation

Create and activate a Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

For GPU-enabled PyTorch installations, the appropriate PyTorch build should be installed for the target machine's CUDA/GPU configuration.

## Dataset

The dataset is intentionally **not included in this repository** because it contains a large collection of images.

The local dataset is expected to be available separately and configured through:

```text
src/data/config.py
```

The repository therefore contains the code required to process the dataset without committing the complete image dataset to Git.

## Running the Project

### Hyperparameter optimization

The Optuna study is configured through:

```text
src/training/study.py
```

The objective function used by Optuna is implemented in:

```text
src/training/objective.py
```

Run the study using the project's configured entry point.

### Final evaluation

The final evaluation entry point is:

```text
src/final_evaluation/main.py
```

This performs the complete test evaluation and logs the final results to MLflow.

## Experiment Reproducibility

The project separates source-code version control from generated experiment artifacts.

Git tracks:

- source code
- dependency specification
- project documentation

Generated/local resources such as datasets, MLflow artifacts, experiment databases, checkpoints, and Python cache files are excluded through `.gitignore`.

This keeps the Git repository lightweight while allowing MLflow to manage experiment and model artifacts.

## Experiment Management

The project uses two complementary systems:

### Git

Git is responsible for tracking:

```text
Source code
Project configuration
Documentation
Dependency specification
```

### MLflow

MLflow is responsible for tracking:

```text
Experiments
Hyperparameters
Training metrics
Evaluation metrics
Model artifacts
Registered models
```

This separation allows the source code and experiment/model artifacts to be managed independently.

## Current Project Milestone

This repository represents the completion of the initial body-measurement regression pipeline, including:

- CNN model development
- hyperparameter optimization
- checkpoint recovery
- validation-based best-model selection
- final test evaluation
- MLflow experiment tracking
- MLflow model registration
