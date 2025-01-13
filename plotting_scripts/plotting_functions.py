import numpy as np
import pyvista as pv
import os
import argparse


def vis(arrays):
    """
    Visualizes one or more NumPy arrays using Matplotlib.

    Parameters:
    arrays (numpy.ndarray or list of numpy.ndarray): 
        A single NumPy array or a list of arrays to visualize.
    """
    # Ensure arrays is a list of arrays
    if isinstance(arrays, np.ndarray):
        arrays = [arrays]
    elif not isinstance(arrays, list) or not all(isinstance(a, np.ndarray) for a in arrays):
        raise ValueError("Input must be a NumPy array or a list of NumPy arrays.")

    num_plots = len(arrays)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]

    for i, (array, ax) in enumerate(zip(arrays, axes)):
        ndim = array.ndim

        if ndim == 1:  # 1D array
            ax.plot(np.transpose(array), marker='o', linestyle='-', color='b')
            ax.set_title(f"1D Array {i + 1}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.grid(True)
        elif ndim == 2:  # 2D array
            im = ax.imshow(np.transpose(array), cmap='viridis', aspect='auto')
            fig.colorbar(im, ax=ax, label="Value")
            ax.set_title(f"2D Array {i + 1}")
            # Add cell borders
            ax.set_xticks(np.arange(-0.5, array.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, array.shape[0], 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", size=0)  # Hide minor ticks
        else:  # Higher dimensions
            ax.set_title(f"Unsupported Array {i + 1}")
            ax.text(0.5, 0.5, "Unsupported Array", 
                    ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.show()

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Visualize a .npy file and save as an image.")
    parser.add_argument("input_file", type=str, help="Path to the .npy file")
    parser.add_argument("output_image", type=str, help="Path to save the output image")
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"File not found: {args.input_file}")

    # Load the Ez data from the .npy file
    ez_data = np.load(args.input_file)

    # Define the spatial grid (assuming consistent grid size)
    nx, ny = ez_data.shape
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

    # Add the Ez data to the grid
    grid["Ez"] = ez_data.ravel()
    warped_grid = grid.warp_by_scalar("Ez", factor=5)  # Adjust the factor

    # PyVista plotter setup
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        warped_grid,
        cmap="viridis",
        clim=[-0.01, 0.01]  # Adjust color limits as needed
    )
    plotter.remove_scalar_bar()

    # Plot sensors as dots on the grid
    sensor_positions = [(x[s[0], s[1]], y[s[0], s[1]], 0) for s in sensors]
    sensor_points = pv.PolyData(sensor_positions)
    plotter.add_points(
        sensor_points,
        color="red",
        point_size=10,
        render_points_as_spheres=True
    )

    plotter.add_axes()
    plotter.set_background("white")

    # Save the output image
    plotter.show(screenshot=args.output_image)
    print(f"Image saved as {args.output_image}")

    
if __name__ == "__main__":
    main()
