import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = "clean.csv"  # Ensure this file exists
df = pd.read_csv(file_path)

# Compute the correlation matrix
correlation_matrix = df.corr()

# Save correlation matrix as a CSV file
correlation_matrix.to_csv("correlation_matrix.csv")
print("Correlation matrix saved as 'correlation_matrix.csv'")

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="Blues", fmt=".2f", linewidths=0.5)

# Save the heatmap as an image
heatmap_file = "correlation_heatmap.png"
plt.title("Correlation Matrix Heatmap")
plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")  # High-quality save
plt.close()  # Close plot to free memory

print(f"Heatmap saved as '{heatmap_file}'")

# Find the highest correlation (excluding self-correlation of 1.0)
corr_matrix_abs = correlation_matrix.abs().copy()
corr_matrix_abs.values[np.tril_indices_from(corr_matrix_abs)] = 0  # Ignore lower triangle & diagonal
max_corr = corr_matrix_abs.max().max()
max_corr_pair = np.where(corr_matrix_abs == max_corr)
row, col = max_corr_pair[0][0], max_corr_pair[1][0]
max_corr_feature1, max_corr_feature2 = correlation_matrix.index[row], correlation_matrix.columns[col]

# Highlight the highest correlation in a new heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(correlation_matrix, cmap="Blues", fmt=".2f", linewidths=0.5)
ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='red', lw=3))

# Save the highlighted heatmap as an image
highlighted_heatmap_file = "correlation_heatmap_highlighted.png"
plt.title(f"Highest Correlation: {max_corr_feature1} & {max_corr_feature2} ({max_corr:.2f})")
plt.savefig(highlighted_heatmap_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Highlighted heatmap saved as '{highlighted_heatmap_file}'")


print(f"Highlighted heatmap saved as '{highlighted_heatmap_file}'")

# Print top 5 correlated features with the target variable (assumed last column)
target_column = df.columns[-1]  # Assuming last column is the target
if target_column in correlation_matrix:
    top_correlated = correlation_matrix[target_column].abs().sort_values(ascending=False)[1:6]  # Exclude self-correlation
    print(f"Top 5 correlated features with '{target_column}':")
    print(top_correlated)
    top_correlated.to_csv("top_5_correlated.csv")
    print("Top 5 correlated features saved as 'top_5_correlated.csv'")