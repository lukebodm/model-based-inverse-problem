import numpy as np
import pyvista as pv
import os
from glob import glob

# Directory containing the Ez .npy files
data_dir = "../plotting_matrices"
output_gif = "Ez_animation.gif"

# Load all .npy files
npy_files = sorted(glob(os.path.join(data_dir, "Ez_timestep_*.npy")))

# Check if any files are found
if not npy_files:
    raise FileNotFoundError(f"No .npy files found in {data_dir}")

# Define the spatial grid (assuming consistent grid size for all files)
sample_data = np.load(npy_files[0])
nx, ny = sample_data.shape
x = np.linspace(0, 1, nx)  # Adjust domain size if necessary
y = np.linspace(0, 1, ny)
x, y = np.meshgrid(x, y)

# Define sensor locations
sensors = [
    (31, 31), (31, 81), (31, 141), (31, 211), (31, 251), (31, 319), 
    (71, 329),  (151, 329), (241, 329), (281, 329), (322, 329),
    (329, 291), (329, 231), (329, 121), (329, 95),  (329, 41),
    (265, 31),  (201, 31),  (101, 31),  (64, 31)
]

# Create a PyVista structured grid
grid = pv.StructuredGrid()
grid.points = np.c_[x.ravel(), y.ravel(), np.zeros_like(x.ravel())]
grid.dimensions = nx, ny, 1

# PyVista plotter setup
plotter = pv.Plotter(off_screen=True)  # Use off_screen for GIF creation
plotter.open_gif(output_gif)

# Add mesh and configure the plotter
for file in npy_files:
    ez_data = np.load(file)  # Load the current Ez field
    grid["Ez"] = ez_data.ravel()  # Update scalar field
    warped_grid = grid.warp_by_scalar("Ez", factor=5)  # Adjust the factor

    # Update the plotter with the new data
    plotter.add_mesh(
        warped_grid,
        cmap="viridis",
        #scalar_bar_args={"title": "Ez Field"},
        clim=[-0.01, 0.01]  # Keep colorbar consistent
    )
    
    # Plot sensors as dots on the grid
    sensor_positions = [(x[s[0], s[1]], y[s[0], s[1]], 0) for s in sensors]
    sensor_points = pv.PolyData(sensor_positions)
    plotter.add_points(
        sensor_points,
        color="red",
        point_size=10,
        render_points_as_spheres=True
    )

    #plotter.add_axes()
    plotter.remove_scalar_bar()
    plotter.set_background("white")

    # Write the frame to the GIF
    plotter.write_frame()

    # Clear the plotter for the next frame
    plotter.clear()

# Close the GIF writer
plotter.close()

print(f"Animation saved as {output_gif}")
