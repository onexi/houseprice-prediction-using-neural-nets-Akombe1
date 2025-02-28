import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Load the CSV file
input_file = "data.csv"  # Replace with your actual file name
df = pd.read_csv(input_file)

# Specify the target column
target_column = "SalePrice"  # Replace with the actual target column name

# Ensure the target column exists
if target_column not in df.columns:
    raise ValueError(f"The target column '{target_column}' is not present in the dataset.")

# Separate features (X) and target (y)
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable

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
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Adjust batch size as needed

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Save processed data as CSV (for reference)
train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df[target_column] = y_train.values
train_df.to_csv("processed_train.csv", index=False)

test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df[target_column] = y_test.values
test_df.to_csv("processed_test.csv", index=False)

print("Processed training data saved as 'processed_train.csv'")
print("Processed testing data saved as 'processed_test.csv'")
print("PyTorch DataLoader ready for training with batch size of 32.")
