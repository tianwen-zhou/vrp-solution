import pandas as pd
import numpy as np

# Step 1: Load the CSV file into a DataFrame
file_path = 'data_jilin.csv'  # Replace with your actual file path
df = pd.read_csv(file_path, delimiter='\t')  # Adjust delimiter if needed, assuming tab-separated

# Step 2: Function to randomly delete rows and return a smaller dataset
def delete_random_rows(df, num_rows_to_keep):
    num_rows_to_delete = len(df) - num_rows_to_keep
    rows_to_delete = np.random.choice(df.index, num_rows_to_delete, replace=False)
    df_reduced = df.drop(rows_to_delete)
    return df_reduced

# Step 3: Create datasets with 10,000, 5,000, and 1,000 rows
df_10000 = delete_random_rows(df, 10000)
df_5000 = delete_random_rows(df, 5000)
df_1000 = delete_random_rows(df, 1000)

# Step 4: Save the datasets to new CSV files
df_10000.to_csv('reduced_data_10000.csv', index=False)
df_5000.to_csv('reduced_data_5000.csv', index=False)
df_1000.to_csv('reduced_data_1000.csv', index=False)

print("Files have been saved:")
print("reduced_data_10000.csv")
print("reduced_data_5000.csv")
print("reduced_data_1000.csv")
