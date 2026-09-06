import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from src.data.bodym_dataset import Target_columns

from src.data.config import Root_dir

checkpoint_dir = Root_dir/f"checkpoints"
scaler_path = checkpoint_dir/f"preprocessed_target.pkl"

Target_column = [
    *Target_columns
]

def preprocess_target(
        train_csv: str,
        val_csv: str,
        test_csv: str
    ):

    # Load the raw data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Extract the target values
    y_train = train_df[Target_column].values
    y_val = val_df[Target_column].values
    y_test = test_df[Target_column].values

    # Performing scalar operation
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    y_test_scaled = scaler.transform(y_test)

    # Save preprocessed scalar target
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, scaler_path)

    print(f"Scaler saved to: {scaler_path}")

    return (
        scaler,
        y_train_scaled.astype(np.float32),
        y_val_scaled.astype(np.float32),
        y_test_scaled.astype(np.float32)
    )