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

sensors = [
    (76, 31), (46, 329), (299, 329), (329, 254), (329, 269), (120, 31),
    (329, 31), (329, 150), (254, 31), (135, 31), (329, 165), (329, 284),
    (91, 31), (61, 329), (314, 329), (195, 329), (329, 46), (31, 120),
    (329, 61), (329, 180), (76, 329), (329, 299), (329, 314), (31, 135),
    (225, 31), (269, 31), (31, 254), (150, 31), (329, 76), (329, 195),
    (31, 269), (106, 31), (329, 210), (120, 329), (240, 31), (31, 31),
    (210, 329), (254, 329), (31, 150), (329, 91), (31, 165), (31, 284),
    (91, 329), (135, 329), (329, 225), (31, 46), (284, 31), (165, 31),
    (31, 61), (31, 299), (329, 240), (31, 314), (46, 31), (299, 31),
    (225, 329), (269, 329), (31, 76), (106, 329), (31, 329), (240, 329),
    (31, 91), (31, 106), (31, 225), (61, 31), (314, 31), (284, 329), (31, 240)
]


# Create a PyVista structured grid
grid = pv.StructuredGrid()
grid.points = np.c_[x.ravel(), y.ravel(), np.zeros_like(x.ravel())]
grid.dimensions = nx, ny, 1

# PyVista plotter setup
plotter = pv.Plotter(off_screen=True)  # Use off_screen for GIF creation
plotter.open_gif(output_gif)

# Add mesh and configure the plotter
for timestep, file in enumerate(npy_files):
    ez_data = np.load(file)  # Load the current Ez field
    grid["Ez"] = ez_data.ravel()  # Update scalar field
    warped_grid = grid.warp_by_scalar("Ez", factor=5)  # Adjust the factor

    # Update the plotter with the new data
    plotter.add_mesh(
        warped_grid,
        cmap="viridis",
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

    # Add the timestep number as text to the figure
    plotter.add_text(f"Timestep: {timestep}", position=(0.5, 1), font_size=20, color="black")

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
