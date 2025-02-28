import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 1. Define the MLP Model (Ensure This is Defined First!)
# -----------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression output
        return x

# -----------------------
# 2. Load the CSV file
# -----------------------
input_file = "clean.csv"
df = pd.read_csv(input_file)

# Specify the target column
target_column = "SalePrice"

# Ensure the target column exists
if target_column not in df.columns:
    raise ValueError(f"The target column '{target_column}' is not present in the dataset.")

# Fill missing values in 'MasVnrArea' with its mean
#df["MasVnrArea"].fillna(df["MasVnrArea"].mean(), inplace=True)
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].mean())


# Separate features (X) and target (y)
X = df.drop(columns=[target_column])  # Features
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(df[[target_column]]).flatten()

# Keep only numeric features
X = X.select_dtypes(include=["number"])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (mean = 0, std = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# -----------------------
# 3. Define Model Parameters
# -----------------------
input_size = X_train.shape[1]  # Dynamically get the number of features
hidden_size = 128
output_size = 1  # Regression output

# -----------------------
# 4. Save Everything to Be Used in Predictions
# -----------------------
# This allows predictions.py to import everything it needs from normalize.py
__all__ = ["MLP", "X_test_tensor", "y_test_tensor", "input_size", "hidden_size", "output_size"]
