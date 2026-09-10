Absolutely. Since you’ve now updated `requirements.txt`, here is a **complete, professional README** that reflects the current state of your project, including the CNN regression model, Optuna, checkpoint recovery, MLflow, model registry, DeepLabV3 segmentation, and FastAPI serving.

# Body Measurement Regression

A PyTorch-based computer vision regression system that predicts **14 human body measurements** from paired **front-view and side-view images**.

The project combines deep learning, hyperparameter optimization, experiment tracking, model registry, human silhouette segmentation, and production model serving through a FastAPI REST API.

---

## Project Overview

The system is designed to predict human body measurements from two complementary views of a person:

- Front-view image
- Side-view image

During model development, the regression model is trained using **binary human silhouette masks** rather than raw RGB photographs.

For production inference, the FastAPI service accepts normal RGB images, automatically extracts the human silhouette from each image using **DeepLabV3-ResNet50**, converts the silhouettes into the same two-channel representation used during training, and passes them to the trained regression model.

### End-to-End Architecture

```text
                    TRAINING PIPELINE

Front-view image ──┐
                   ├──> Preprocessing ──> 2-channel input
Side-view image ───┘                         │
                                            ▼
                                      CNN Regression Model
                                            │
                                            ▼
                                     14 Body Measurements
                                            │
                                            ▼
                                    Validation / Testing
                                            │
                                            ▼
                                    Optuna Optimization
                                            │
                                            ▼
                                      Best Model
                                            │
                                            ▼
                                         MLflow
                                            │
                                            ▼
                                    Model Registry
```

```text
                    PRODUCTION PIPELINE

Front RGB image ──> DeepLabV3-ResNet50 ──> Front binary mask ──┐
                                                               │
                                                               ├──> Preprocessing
                                                               │
Side RGB image ───> DeepLabV3-ResNet50 ──> Side binary mask ──┘
                                                                    │
                                                                    ▼
                                                            2-channel tensor
                                                                    │
                                                                    ▼
                                                         BodyRegressionModel
                                                                    │
                                                                    ▼
                                                          14 body measurements
```

---

# Key Features

- PyTorch CNN regression model
- Two-view input using front and side human silhouettes
- Four convolutional blocks
- Batch Normalization
- ReLU activation
- Max Pooling
- Adaptive Average Pooling
- Configurable fully connected regression head
- Dropout regularization
- Optuna hyperparameter optimization
- SGD, Adam, and AdamW optimizer search
- ReduceLROnPlateau learning-rate scheduling
- MSE regression loss
- Crash-recovery checkpoints
- Global best-model checkpointing
- MLflow experiment tracking
- MLflow Model Registry
- Final test-set evaluation
- MAE, MSE, and RMSE metrics
- DeepLabV3-ResNet50 human segmentation
- Binary human silhouette generation
- FastAPI REST API
- Multipart image uploads
- Production inference pipeline

---

# Project Structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt
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
    ├── serving/
    │   ├── app.py
    │   ├── model_loader.py
    │   ├── prediction.py
    │   ├── preprocessing.py
    │   ├── schema.py
    │   └── segmentation.py
    │
    └── utils/
        ├── __init__.py
        └── preprocessing.py
```

---

# Directory Responsibilities

## `src/data/`

Responsible for dataset configuration, dataset implementation, and PyTorch DataLoader creation.

### `bodym_dataset.py`

Contains the dataset implementation used to load paired front-view and side-view body images together with their target measurements.

### `config.py`

Contains dataset and data-processing configuration.

### `dataloader.py`

Creates the training, validation, and test DataLoaders.

The model receives input tensors with the general structure:

```text
[batch_size, 2, 224, 224]
```

where:

```text
2 = front-view channel + side-view channel
```

The target tensor has the structure:

```text
[batch_size, 14]
```

because the model predicts 14 measurements.

---

# Model Architecture

The main regression model is implemented in:

```text
src/models/model.py
```

The model class is:

```python
BodyRegressionModel
```

## Convolutional Backbone

The CNN contains four convolutional blocks.

### Block 1

```text
Conv2d(2 → 32)
BatchNorm
ReLU
MaxPool
```

### Block 2

```text
Conv2d(32 → 64)
BatchNorm
ReLU
MaxPool
```

### Block 3

```text
Conv2d(64 → 128)
BatchNorm
ReLU
MaxPool
```

### Block 4

```text
Conv2d(128 → 256)
BatchNorm
ReLU
```

The convolutional layers use:

```text
Kernel size = 3 × 3
Padding = 1
```

The first three convolutional blocks use max pooling to progressively reduce spatial dimensions while increasing the number of learned feature channels.

The final convolutional block produces 256 feature maps.

---

# Adaptive Average Pooling

After the convolutional blocks, the model uses:

```text
AdaptiveAvgPool2d((1, 1))
```

This reduces each feature map to a single value.

The resulting representation is flattened before being passed into the regression head.

Adaptive pooling allows the architecture to produce a fixed-size representation regardless of small spatial-size changes before the pooling operation.

---

# Regression Head

The fully connected regression head is configurable.

Its structure is approximately:

```text
Flatten
   │
   ▼
Linear
   │
   ▼
ReLU
   │
   ▼
Dropout
   │
   ▼
Linear
   │
   ▼
ReLU
   │
   ▼
Dropout
   │
   ▼
Linear
   │
   ▼
14 measurements
```

The following parameters are configurable:

- `dropout`
- `hidden_layers`

The final linear layer produces exactly:

```text
14 outputs
```

---

# Training

The training loop is implemented in:

```text
src/training/training.py
```

Validation is handled in:

```text
src/training/evaluate.py
```

The regression loss function is:

```python
nn.MSELoss()
```

The model is trained to minimize validation loss while Optuna searches for an effective combination of hyperparameters.

---

# Hyperparameter Optimization

Optuna is used to search for the best training configuration.

The optimization logic is implemented in:

```text
src/training/objective.py
```

and the Optuna study is managed through:

```text
src/training/study.py
```

## Search Space

### Dropout

```text
0.2 – 0.5
```

### Hidden Layers

Possible values:

```text
32
64
128
256
```

### Learning Rate

Log-uniform search:

```text
5e-6 – 1e-3
```

### Batch Size

```text
32
64
```

### Optimizer

The study evaluates:

```text
SGD
Adam
AdamW
```

Optimizer-specific hyperparameters are also searched.

### SGD Momentum

```text
0.8 – 0.99
```

### AdamW Weight Decay

Log-uniform search:

```text
1e-6 – 1e-2
```

---

# Learning Rate Scheduler

The training process uses:

```python
torch.optim.lr_scheduler.ReduceLROnPlateau
```

with:

```text
mode = "min"
factor = 0.5
patience = 3
```

The scheduler monitors validation loss.

When validation loss stops improving for the configured patience period, the learning rate is reduced.

This allows training to take larger optimization steps initially and smaller steps when the model approaches a better solution.

---

# Checkpoint Management

Checkpoint functionality is implemented in:

```text
src/training/checkpoint.py
```

The project uses checkpointing for two different purposes.

## 1. Crash-Recovery Checkpoint

The temporary checkpoint allows training to resume if the computer crashes, the process is interrupted, or training otherwise stops unexpectedly.

It stores information such as:

```text
trial number
trial parameters
epoch
model state
optimizer state
scheduler state
training loss
validation loss
best validation loss
```

A recovery checkpoint should only be resumed when it belongs to the same compatible trial/configuration.

Temporary recovery checkpoints are removed after the trial completes or is pruned.

---

## 2. Global Best-Model Checkpoint

The project separately tracks the best model found across Optuna trials.

The global best model is updated only when a trial achieves a validation loss better than the previous global best.

The checkpoint contains information such as:

```text
trial_number
trial_params
epoch
best_val_loss
model_state_dict
optimizer_state_dict
scheduler_state_dict
```

This separation is important because:

```text
Crash checkpoint
        ↓
Purpose: resume interrupted training

Best-model checkpoint
        ↓
Purpose: preserve the best model discovered by the experiment
```

They solve different problems and therefore should not be treated as the same checkpoint.

---

# MLflow Experiment Tracking

MLflow is used to track experiments, hyperparameters, metrics, artifacts, and models.

Tracked information includes:

- Trial number
- Dropout
- Hidden-layer size
- Learning rate
- Batch size
- Optimizer
- Optimizer-specific parameters
- Training loss
- Validation loss
- Trial best validation loss
- Global best validation loss
- Learning rate

The MLflow tracking server can be started with:

```powershell
mlflow ui
```

The local MLflow interface is typically available at:

```text
http://127.0.0.1:5000
```

---

# MLflow Model Registry

After final evaluation, the trained model can be logged to MLflow and registered in the MLflow Model Registry.

The final evaluation components are located in:

```text
src/final_evaluation/
```

The registry workflow allows the trained model to move from experimentation toward a reusable production artifact.

---

# Final Model Evaluation

Final evaluation is implemented in:

```text
src/final_evaluation/
```

The main entry point is:

```text
src/final_evaluation/main.py
```

The evaluation pipeline performs the following operations:

```text
Load best model checkpoint
        ↓
Retrieve best-trial hyperparameters
        ↓
Reconstruct model architecture
        ↓
Create test DataLoader
        ↓
Load target scaler
        ↓
Run predictions
        ↓
Inverse-transform predictions
        ↓
Calculate metrics
        ↓
Generate evaluation report
        ↓
Log results to MLflow
        ↓
Register model
```

---

# Evaluation Metrics

The final model is evaluated using:

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual measurements.

### MSE

Mean Squared Error penalizes larger prediction errors more heavily.

### RMSE

Root Mean Squared Error is the square root of MSE and expresses the error in the same units as the target measurements.

Metrics are calculated both overall and per body measurement where applicable.

---

# Human Silhouette Segmentation

The production serving pipeline introduces an additional computer vision stage.

The regression model was trained on **binary human silhouettes**, but users of the API provide normal RGB photographs.

Therefore, RGB images cannot simply be passed directly to the regression model.

Instead, each image first passes through a human segmentation model.

The segmentation implementation is located at:

```text
src/serving/segmentation.py
```

The segmentation model is:

```text
DeepLabV3-ResNet50
```

from `torchvision`.

---

# Segmentation Pipeline

For each uploaded image:

```text
RGB image
    │
    ▼
DeepLabV3-ResNet50
    │
    ▼
Human segmentation prediction
    │
    ▼
Binary mask
    │
    ├── 0   = background
    │
    └── 255 = human silhouette
```

Both front and side images go through this process.

```text
Front RGB image ──> Segmentation ──> Front binary mask
Side RGB image  ──> Segmentation ──> Side binary mask
```

The two binary masks are then combined into the two-channel input expected by the regression model.

---

# FastAPI Model Serving

The production API is implemented in:

```text
src/serving/
```

The main application is:

```text
src/serving/app.py
```

The API exposes the trained body-measurement model through HTTP.

---

# Serving Components

## `app.py`

Defines the FastAPI application and API endpoints.

It handles:

- HTTP requests
- Uploaded images
- Input validation
- Segmentation
- Model preprocessing
- Prediction
- API responses
- HTTP errors

---

## `model_loader.py`

Responsible for loading the trained model from the configured model source/checkpoint and reconstructing the correct model architecture.

The model is moved to the configured device:

```python
torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

and placed into evaluation mode for inference.

---

## `segmentation.py`

Contains the human segmentation component.

It loads DeepLabV3-ResNet50 and converts an RGB human photograph into a binary human silhouette mask.

---

## `preprocessing.py`

Converts the generated front and side masks into the representation expected by the regression model.

The final input has the form:

```text
[1, 2, 224, 224]
```

where:

```text
1 = batch dimension
2 = front + side channels
224 × 224 = image dimensions
```

---

## `prediction.py`

Responsible for running inference through the trained regression model.

It handles the model input and prediction process and converts the model output into the API response structure.

---

## `schema.py`

Defines the Pydantic response schemas used by FastAPI.

This provides structured and validated API responses.

---

# FastAPI Prediction Endpoint

The primary endpoint is:

```text
POST /predict
```

It accepts two image files:

```text
front_image
side_image
```

Supported image formats are:

```text
JPEG
PNG
WEBP
```

The request uses:

```text
multipart/form-data
```

---

# Prediction Flow

A request to `/predict` follows this sequence:

```text
Client
  │
  │ front_image + side_image
  ▼
FastAPI
  │
  ▼
Validate file types
  │
  ▼
Load images
  │
  ├───────────────┐
  ▼               ▼
Front image     Side image
  │               │
  ▼               ▼
DeepLabV3       DeepLabV3
  │               │
  ▼               ▼
Front mask      Side mask
  │               │
  └───────┬───────┘
          ▼
     Preprocessing
          │
          ▼
   2-channel tensor
          │
          ▼
 BodyRegressionModel
          │
          ▼
 14 measurements
          │
          ▼
 FastAPI response
```

This ensures that the production inference input is consistent with the data representation used to train the regression model.

---

# Running the FastAPI Server

From the project root:

```powershell
uvicorn src.serving.app:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation.

The Swagger UI is available at:

```text
http://127.0.0.1:8000/docs
```

---

# API Usage

Send a `POST` request to:

```text
/predict
```

using multipart form-data.

Required fields:

```text
front_image = front-view photograph
side_image  = side-view photograph
```

Example request structure:

```text
POST http://127.0.0.1:8000/predict

Content-Type: multipart/form-data

front_image: <front image>
side_image:  <side image>
```

The server performs segmentation and regression automatically.

---

# Device Support

The project automatically selects CUDA when a compatible NVIDIA GPU is available:

```python
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Otherwise, it falls back to CPU.

The development environment was tested with:

```text
GPU: NVIDIA GeForce GTX 1050
CUDA: 12.6
PyTorch: 2.12.1+cu126
```

The project therefore supports both:

```text
CUDA GPU
CPU
```

depending on the available environment.

---

# Installation

## 1. Clone the Repository

```powershell
git clone https://github.com/Preciousboy/body-measurement-regression.git
```

Navigate into the project:

```powershell
cd body-measurement-regression
```

---

## 2. Create a Virtual Environment

```powershell
py -3.14 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

Install all project dependencies from:

```text
requirements.txt
```

using:

```powershell
pip install -r requirements.txt
```

The dependency file contains the project's pinned package versions and serving dependencies.

---

# Main Dependencies

The project uses the following major technologies:

| Technology       | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| PyTorch          | Deep learning and CNN regression                  |
| Torchvision      | Computer vision models and DeepLabV3 segmentation |
| NumPy            | Numerical computation                             |
| Pandas           | Data manipulation                                 |
| Scikit-learn     | Preprocessing and evaluation                      |
| Joblib           | Serialization of preprocessing objects            |
| tqdm             | Training progress visualization                   |
| Optuna           | Hyperparameter optimization                       |
| MLflow           | Experiment tracking and model registry            |
| FastAPI          | Production model-serving API                      |
| Uvicorn          | ASGI server for FastAPI                           |
| python-multipart | Multipart image upload support                    |
| Pillow           | Image loading and processing                      |
| Cloudpickle      | Serialization support                             |

The exact versions are maintained in:

```text
requirements.txt
```

---

# Dataset

The dataset contains paired front-view and side-view images together with body measurement targets.

The dataset is **not included in this repository**.

Dataset configuration is handled through:

```text
src/data/config.py
```

The target preprocessing scaler is stored separately from the source code and is used during inference to convert predictions back into the original measurement scale.

---

# Input Representation

The regression model does not directly consume ordinary RGB photographs.

Instead, its input consists of two silhouette channels:

```text
Channel 0 → Front-view silhouette
Channel 1 → Side-view silhouette
```

The resulting tensor has the structure:

```text
[batch_size, 2, 224, 224]
```

This representation allows the CNN to learn complementary geometric information from the person's front and side views.

---

# Production Input vs Training Input

An important architectural distinction exists between training and production.

### During training

```text
Prepared silhouette masks
        ↓
Preprocessing
        ↓
2-channel tensor
        ↓
Regression model
```

### During production

```text
RGB photographs
        ↓
DeepLabV3 segmentation
        ↓
Binary silhouette masks
        ↓
Preprocessing
        ↓
2-channel tensor
        ↓
Regression model
```

The segmentation stage bridges the gap between real-world user input and the representation expected by the trained regression model.

---

# Reproducibility

The repository uses Git to track:

- Source code
- Configuration
- Documentation
- Dependency specifications

Large or generated artifacts are excluded from version control.

These include:

- Dataset files
- MLflow artifacts
- Experiment databases
- Checkpoints
- Python cache files
- Temporary recovery files

The `.gitignore` file contains the appropriate exclusions.

---

# Experiment Management

The project separates source-code versioning from experiment management.

### Git

Git is responsible for:

```text
Source code
Configuration
Documentation
Dependency specification
```

### MLflow

MLflow is responsible for:

```text
Experiments
Hyperparameters
Training metrics
Validation metrics
Artifacts
Models
Model versions
```

### Optuna

Optuna is responsible for:

```text
Hyperparameter search
Trial management
Trial ranking
Pruning
Best-trial selection
```

This separation keeps the development workflow organized and reproducible.

---

# Development Workflow

The complete development workflow is:

```text
Dataset
   │
   ▼
Data preprocessing
   │
   ▼
PyTorch DataLoader
   │
   ▼
CNN development
   │
   ▼
Training
   │
   ▼
Validation
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
MLflow tracking
   │
   ▼
MLflow Model Registry
   │
   ▼
FastAPI serving
   │
   ▼
RGB image upload
   │
   ▼
DeepLabV3 human segmentation
   │
   ▼
Silhouette preprocessing
   │
   ▼
Body measurement prediction
```

---

# Current Project Milestone

The project currently includes:

- CNN-based body measurement regression
- Four-block convolutional architecture
- Batch normalization
- Dropout regularization
- Adaptive average pooling
- Configurable regression head
- Optuna hyperparameter optimization
- Multiple optimizer search
- Learning-rate scheduling
- Crash-recovery checkpointing
- Global best-model checkpointing
- Validation-based model selection
- Final test-set evaluation
- MAE, MSE, and RMSE evaluation
- MLflow experiment tracking
- MLflow Model Registry
- DeepLabV3-ResNet50 human segmentation
- Binary human silhouette generation
- Front/side two-channel preprocessing
- FastAPI model serving
- Multipart image upload
- Production inference pipeline

---

# Technology Stack

```text
Python
    │
    ├── PyTorch
    │     └── CNN Regression
    │
    ├── Torchvision
    │     └── DeepLabV3-ResNet50
    │
    ├── Optuna
    │     └── Hyperparameter Optimization
    │
    ├── MLflow
    │     ├── Experiment Tracking
    │     └── Model Registry
    │
    ├── FastAPI
    │     └── REST API
    │
    └── Uvicorn
          └── API Server
```

---

# Repository

GitHub repository:

```text
https://github.com/Preciousboy/body-measurement-regression
```

---

# Project Goal

The long-term goal of this project is to develop a complete computer vision system capable of transforming ordinary human photographs into useful body measurement predictions through an automated machine-learning pipeline.

The project therefore covers the complete lifecycle of a machine-learning system:

```text
Data
 ↓
Preprocessing
 ↓
Model Development
 ↓
Training
 ↓
Hyperparameter Optimization
 ↓
Experiment Tracking
 ↓
Model Selection
 ↓
Evaluation
 ↓
Model Registry
 ↓
Computer Vision Preprocessing
 ↓
API Deployment
 ↓
Production Inference
```

This makes the project more than a standalone neural network: it represents an end-to-end machine-learning system from model development through production serving.

This version is ready to replace the current README. **Keep `requirements.txt` as the authoritative source for exact package versions**, while the README explains what each dependency and component does.
