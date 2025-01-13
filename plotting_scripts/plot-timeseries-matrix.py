import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file into a pandas DataFrame
df = pd.read_csv('example_training_data.csv')

# Step 2: Exclude the first 4 columns and transpose the matrix
data = df.iloc[:, 4:].values.T  # Transpose the matrix using `.T`

# Step 3: Create a heatmap using pcolormesh
plt.figure(figsize=(10, 6))  # Optional: adjust the size of the plot

# Create a pcolormesh plot with edges
plt.pcolormesh(data, cmap='viridis', edgecolors='k', linewidth=0.5)

# Step 4: Label the axes
plt.xlabel('Row Index (Original Data)')
plt.ylabel('Column Index (Excluding First 4)')

# Step 5: Show the plot
plt.title('Transposed Heatmap of Training Data (Excluding First 4 Columns)')
plt.show()
