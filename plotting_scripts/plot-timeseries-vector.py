import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the CSV file into a pandas DataFrame
df = pd.read_csv('example_training_data.csv')

# Step 2: Exclude the first 4 columns and stack all columns into one column
data = df.iloc[:, 4:].values.ravel()  # Flatten the data into a 1D vector
data = data.reshape(-1, 1)  # Reshape into a 2D matrix with one column

# Step 3: Create a heatmap using pcolormesh
plt.figure(figsize=(2, 12))  # Adjust the size of the plot for a tall, narrow view

# Create a pcolormesh plot with edges
plt.pcolormesh(data, cmap='viridis', edgecolors='k', linewidth=0)

# Step 4: Label the axes
plt.xlabel('Single Column')
plt.ylabel('Index')

# Step 5: Show the plot
plt.title('Flattened Heatmap of Training Data (Stacked Columns)')
plt.show()
