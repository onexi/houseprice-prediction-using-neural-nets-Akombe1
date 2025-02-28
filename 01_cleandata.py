import pandas as pd

def clean_data(input_file, target_column,numb, missing_threshold=50):
    # Load the data from the CSV file
    data = pd.read_csv(input_file)

    # Print the original number of rows and columns
    print(f"Original number of rows: {data.shape[0]}")
    print(f"Original number of columns: {data.shape[1]}")

    # Ensure the target column is present
  #  if target_column not in data.columns:
    #    raise ValueError(f"The target column '{target_column}' is not present in the dataset.")

    # Identify columns with missing values above the threshold
    columns_to_drop = data.columns[data.isna().sum() >= missing_threshold]

    # Drop those columns
    data_clean = data.drop(columns=columns_to_drop)

    # Select only numeric columns + ensure the target column is retained
    numeric_columns = data_clean.select_dtypes(include=["number"]).columns
    if numb ==0: 
        if target_column not in numeric_columns:
            numeric_columns = numeric_columns.append(pd.Index([target_column]))  # Ensure target column stays

    data_clean = data_clean[numeric_columns]  # Keep only numeric columns

    # Print the number of rows and columns after cleaning
    print(f"Number of rows after cleaning: {data_clean.shape[0]}")
    print(f"Number of columns after cleaning: {data_clean.shape[1]}")

    # Save the cleaned data to a new CSV file
    if numb == 0:
        output_file = "clean.csv"
    if numb == 1:
        output_file = "cleantest.csv"
    
    data_clean.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

# Example usage
input_file = "train.csv"  # Replace with your actual file
input_filetwo = "test.csv"  # Replace with your actual file

target_column = "SalePrice"  # Replace with your actual target column
clean_data(input_file, target_column,0)
clean_data(input_filetwo, target_column,1)