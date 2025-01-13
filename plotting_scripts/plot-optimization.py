import sys
from pathlib import Path
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Add the parent directory to the Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import PinnModel from surrogate_model.py
from surrogate_model import PinnModel

# Initialize the model and load weights
model = PinnModel()
model.load_state_dict(torch.load('../model.pth'))
model.eval()  # Set the model to evaluation mode

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Path to test data
test_data_path = Path("../test_data")

# Regex to extract values from filenames
filename_regex = r"eps(\d+)_mu(\d+)_w(\d+)_l(\d+)\.npz"

# Storage for plot data
results = []

# Process every 100th file
for i, file in enumerate(test_data_path.glob("*.npz")):
    if i % 10 != 0:
        continue  # Skip files that are not every 100th

    match = re.match(filename_regex, file.name)
    if not match:
        print(f"Skipping invalid file name: {file.name}")
        continue

    # Extract values from the filename
    eps = float(match.group(1)) / 1000  # Convert to scientific notation
    mu = float(match.group(2)) / 1000
    w = float(match.group(3))
    l = float(match.group(4))

    # Calculate averages
    avg_eps_mu = (eps + mu) / 2
    avg_w_l = (w + l) / 2

    # Apply constraints
    if not (4 <= avg_eps_mu <= 8 and 100 <= avg_w_l <= 200):
        continue  # Skip files outside the specified range

    # Create input tensor
    input_vector = torch.tensor([[eps, mu, w, l]], dtype=torch.float32).to(device)

    # Run the model
    output = model(input_vector).detach().cpu().numpy().flatten()  # Model output

    # Load ground truth data
    data = np.load(file)
    ground_truth = data["output_data"].flatten()  # Flatten for comparison

    # Calculate L2 norm (loss)
    loss = np.linalg.norm(output - ground_truth)

    # Store results: avg_eps_mu, avg_w_l, loss
    results.append((avg_eps_mu, avg_w_l, loss))

# Convert results to NumPy array
results = np.array(results)

# Grid interpolation for surface plot
x = results[:, 0]  # avg_eps_mu
y = results[:, 1]  # avg_w_l
z = results[:, 2]  # loss

# Define grid for interpolation
grid_x, grid_y = np.meshgrid(
    np.linspace(x.min(), x.max(), 100),
    np.linspace(y.min(), y.max(), 100),
)

# Interpolate z values onto the grid
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# Plot the surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Surface plot
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="viridis", edgecolor="none", alpha=0.8)

# Add color bar

# Label axes with increased font size

# Show plot
plt.show()
