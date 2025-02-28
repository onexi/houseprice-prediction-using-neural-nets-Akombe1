import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load preprocessed data
data = torch.load("preprocessed_data.pth")

X_test_tensor = data["X_test_tensor"]
y_test_tensor = data["y_test_tensor"]
input_size = data["input_size"]
scaler = data["scaler"]  # Feature scaler (if needed for inverse transform)
y_scaler = data["y_scaler"]  # Target variable scaler (to get real price values)

# Define the MLP Model
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size, 128, 1)
model.load_state_dict(torch.load("mlp_model.pth", map_location=device))
model.to(device)
model.eval()

# Ensure tensors are on the correct device
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Make Predictions
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = torch.nn.MSELoss()(predictions, y_test_tensor).item()
    print("Test Mean Squared Error (Torch):", test_loss)

# Convert predictions to original scale
predictions_np = predictions.cpu().numpy().flatten()
y_test_np = y_test_tensor.cpu().numpy().flatten()

# Convert predictions back to original price scale
predictions_original = y_scaler.inverse_transform(predictions_np.reshape(-1, 1)).flatten()
y_test_original = y_scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()

# Compute evaluation metrics
mse = mean_squared_error(y_test_original, predictions_original)
mae = mean_absolute_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

# Print evaluation results
print("Test MSE (Scikit-learn):", mse)
print("Test MAE:", mae)
print("Test R^2 Score:", r2)

# Save Predictions to CSV
results_df = pd.DataFrame({
    "Actual": y_test_original,
    "Predicted": predictions_original
})
results_df.to_csv("model_predictions.csv", index=False)
print("Predictions saved to 'model_predictions.csv'")
