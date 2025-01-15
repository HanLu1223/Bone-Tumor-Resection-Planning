import numpy as np
import trimesh
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_voxels_from_mesh(mesh_file, voxel_size):

    # Load the mesh using trimesh
    mesh = trimesh.load(mesh_file)

    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds

    # Create a grid of points (uniform voxel centers)
    x = np.arange(min_bound[0], max_bound[0] + voxel_size, voxel_size)
    y = np.arange(min_bound[1], max_bound[1] + voxel_size, voxel_size)
    z = np.arange(min_bound[2], max_bound[2] + voxel_size, voxel_size)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    voxel_centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    # Check which voxel centers are inside the mesh (includes the solid part)
    inside = mesh.contains(voxel_centers)

    # Keep only voxels inside the mesh
    inside_voxels = voxel_centers[inside]

    return inside_voxels, voxel_size


def generate_solid_voxels_from_mesh(mesh_file, voxel_size):
    """
    Generate solid voxels inside a 3D mesh.

    Parameters:
        mesh_file (str): Path to the mesh file.
        voxel_size (float): Size of each voxel.

    Returns:
        inside_voxels (np.ndarray): Coordinates of voxel centers inside the mesh.
        voxel_size (float): The size of the voxels used.
    """
    # Load the mesh using trimesh
    mesh = trimesh.load(mesh_file)

    # Get the bounding box of the mesh
    min_bound, max_bound = mesh.bounds

    # Create a grid of points (uniform voxel centers)
    x = np.arange(min_bound[0], max_bound[0] + voxel_size, voxel_size)
    y = np.arange(min_bound[1], max_bound[1] + voxel_size, voxel_size)
    z = np.arange(min_bound[2], max_bound[2] + voxel_size, voxel_size)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    voxel_centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    # Check which voxel centers are inside the mesh
    # This includes solid interior and surface points
    inside = mesh.contains(voxel_centers)

    # Keep only voxels inside the mesh
    inside_voxels = voxel_centers[inside]

    return inside_voxels, voxel_size


def rotate_voxels(voxels, angle, axis='z'):
    """
    Rotate voxel coordinates around a specified axis.

    Parameters:
        voxels (numpy.ndarray): Array of voxel coordinates (N x 3).
        angle (float): Rotation angle in degrees.
        axis (str): Axis to rotate around ('x', 'y', 'z').

    Returns:
        numpy.ndarray: Rotated voxel coordinates.
    """
    # Convert angle to radians
    theta = np.radians(angle)

    # Define rotation matrices
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    # Apply the rotation matrix to all voxel coordinates
    rotated_voxels = np.dot(voxels, rotation_matrix.T)
    return rotated_voxels


def visualize_voxels(voxels, voxel_size):
    """
    Fast and interactive visualization of voxels using PyVista.
    """
    # Calculate the grid dimensions based on the range of voxel coordinates
    grid_dimensions = np.ptp(voxels, axis=0) // voxel_size + 2  # Plus padding for edges
    grid_dimensions = grid_dimensions.astype(int)

    # Create an ImageData object
    grid = pv.ImageData()

    # Define the grid's dimensions, origin, and spacing
    grid.dimensions = tuple(grid_dimensions + 1)  # Add 1 for point-to-cell mapping
    grid.origin = (
        voxels[:, 0].min() - voxel_size / 2,
        voxels[:, 1].min() - voxel_size / 2,
        voxels[:, 2].min() - voxel_size / 2,
    )
    grid.spacing = (voxel_size, voxel_size, voxel_size)

    # Create a binary mask for the voxels
    mask_shape = tuple(grid_dimensions)
    mask = np.zeros(mask_shape, dtype=bool)  # Dimensions for cell data
    indices = ((voxels - grid.origin) / voxel_size).astype(int)
    mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    # Add the mask as cell data
    grid.cell_data["voxels"] = mask.flatten(order="F")

    # Extract the voxels marked as True
    threshold = grid.threshold(0.5, scalars="voxels")

    # Create a PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(threshold, color="blue", show_edges=True, opacity=1)
    plotter.show_grid()
    plotter.show()


# Usage example
mesh_file = "output_simplified.stl"  # Replace with your STL file
voxel_size = 2  # Define the size of each voxel

# Generate and visualize voxels
voxels, voxel_size = generate_solid_voxels_from_mesh(mesh_file, voxel_size)
print(f"Generated {len(voxels)} voxels inside the mesh.")

#visualize_voxels(voxels, voxel_size)
np.save("voxels.npy", voxels)

