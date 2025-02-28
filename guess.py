import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from normalize import MLP  # Import the trained model structure

# Load preprocessed data (scalers from training phase)
data = torch.load("preprocessed_data.pth")
scaler = data["scaler"]  # Feature scaler
y_scaler = data["y_scaler"]  # Target scaler
input_size = data["input_size"]

# Load trained model
model = MLP(input_size, 128, 1)  # Hidden size is 128, output size is 1
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()

# Load new data from train.csv
input_file = "cleantest.csv"
df = pd.read_csv(input_file)

# Remove 'SalePrice' if present (since we are predicting it)
if "SalePrice" in df.columns:
    df = df.drop(columns=["SalePrice"])

# Ensure only numeric columns are used
X = df.select_dtypes(include=["number"])

# Match training feature names (drop extras, add missing ones as 0)
train_feature_names = data["scaler"].feature_names_in_  # Get trained feature names
X = X.reindex(columns=train_feature_names, fill_value=0)  # Ensure order & match columns

# Standardize features using the scaler from training
X_scaled = scaler.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    predictions = model(X_tensor).numpy().flatten()

# Convert predictions back to original scale
predictions_original = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Save predictions to CSV
output_df = df.copy()
output_df["Predicted_SalePrice"] = predictions_original
output_df.to_csv("priceguessed.csv", index=False)

print("Predicted sales prices saved as 'priceguessed.csv'")
