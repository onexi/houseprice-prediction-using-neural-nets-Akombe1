import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 1. Load and Preprocess Data
# -----------------------
input_file = "clean.csv"
df = pd.read_csv(input_file)

# Specify the target column
target_column = "SalePrice"

# Fill missing values
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].mean())

# Separate features (X) and target (y)
X = df.drop(columns=[target_column])
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(df[[target_column]]).flatten()

# Keep only numeric features
X = X.select_dtypes(include=["number"])

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Save processed data to a file
torch.save({
    "X_test_tensor": X_test_tensor,
    "y_test_tensor": y_test_tensor,
    "input_size": X_train.shape[1],
    "scaler": scaler,  # Save the feature scaler
    "y_scaler": y_scaler  # Save the target variable scaler
}, "preprocessed_data.pth")

print("Preprocessed data saved as 'preprocessed_data.pth'")

# -----------------------
# 2. Model Training
# -----------------------
# Define the MLP Model
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

# Model Parameters
hidden_size = 128
output_size = 1  # Regression output

# Initialize Model, Loss Function, and Optimizer
model = MLP(X_train.shape[1], hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True):
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train_tensor):.4f}")

# Save the trained model
torch.save(model.state_dict(), "mlp_model.pth")
print("Model saved as 'mlp_model.pth'")
