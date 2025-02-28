import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from normalize import MLP, input_size, hidden_size, output_size  # Replace 'train' with the actual script filename


# Load the trained model
model = MLP(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()

# Ensure tensors are on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Make predictions
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor).item()
    print("Test Mean Squared Error (Torch):", test_loss)

# Convert predictions to NumPy for evaluation
predictions_np = predictions.cpu().numpy().flatten()
y_test_np = y_test_tensor.cpu().numpy().flatten()

# Compute evaluation metrics
mse = mean_squared_error(y_test_np, predictions_np)
mae = mean_absolute_error(y_test_np, predictions_np)
r2 = r2_score(y_test_np, predictions_np)

# Print evaluation results
print("Test MSE (Scikit-learn):", mse)
print("Test MAE:", mae)
print("Test R^2 Score:", r2)

# Save predictions to CSV for analysis
import pandas as pd
results_df = pd.DataFrame({
    "Actual": y_test_np,
    "Predicted": predictions_np
})
results_df.to_csv("model_predictions.csv", index=False)
print("Predictions saved to 'model_predictions.csv'")
