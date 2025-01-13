import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load the CSV file into a DataFrame
data = pd.read_csv("example_training_data.csv")

# Exclude the first 4 columns
data = data.iloc[:, 4:]

# Get the number of columns to plot
num_columns = data.shape[1]

# Calculate maximum absolute values for all columns
max_abs_values = data.abs().max(axis=0)

# Normalize the values to map to a colormap
norm = plt.Normalize(vmin=max_abs_values.min(), vmax=max_abs_values.max())
colormap = cm.get_cmap('copper')  # Choose a colormap (e.g., viridis)

# Create subplots
fig, axes = plt.subplots(num_columns, 1, figsize=(15, 3 * num_columns), sharex=True)

# Plot each column in its own subplot
for idx, col in enumerate(data.columns):
    # Get the color based on the maximum absolute value
    color = colormap(norm(max_abs_values[col]))
    axes[idx].plot(data[col], label=col, color=color)  # Use the calculated color
    axes[idx].set_xlim(0, 350)  # Limit x-axis to 0–350
    axes[idx].set_yticklabels([])  # Remove y-axis tick labels
    
    # Increase the border line width for each subplot
    for spine in axes[idx].spines.values():
        spine.set_linewidth(2)  # Set the desired linewidth (e.g., 2)
        spine.set_edgecolor('#D3D3D3')  # Set the border color to gray

# Add a common x-axis label
plt.xlabel("Time (row index)", fontsize=14)

# Add a colorbar to indicate the mapping
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])  # Required for colorbar

# Adjust layout to add more vertical space
plt.subplots_adjust(hspace=0.7)  # Increase hspace value for more space between subplots

# Save the plot as a file
plt.savefig("plot_output.png", dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

plt.show()
