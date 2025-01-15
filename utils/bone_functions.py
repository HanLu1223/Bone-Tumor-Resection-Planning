import numpy as np
import random
import pyvista as pv
import trimesh
from scipy.spatial.transform import Rotation as R
import torch

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
    plotter.add_mesh(threshold, color="blue", show_edges=True, opacity=1, label="Bone")
    plotter.add_legend()
    plotter.show_grid()
    plotter.show()

def visualize_two_voxels(voxels1, voxels2, voxel_size):
    """
    Interactive visualization of two voxel sets using PyVista, with different colors for each set.

    Parameters:
        voxels1 (numpy.ndarray): Array of voxel center coordinates for the first voxel set (N x 3).
        voxels2 (numpy.ndarray): Array of voxel center coordinates for the second voxel set (M x 3).
        voxel_size (float): Size of each voxel (assumed cubic for simplicity).
    """
    # Calculate the grid dimensions based on the combined range of both voxel sets
    min_bounds = np.minimum(voxels1.min(axis=0), voxels2.min(axis=0)) - voxel_size / 2
    max_bounds = np.maximum(voxels1.max(axis=0), voxels2.max(axis=0)) + voxel_size / 2
    grid_dimensions = ((max_bounds - min_bounds) // voxel_size + 1).astype(int)

    # Create an empty 3D grid for the first voxel set
    grid1 = np.zeros(grid_dimensions, dtype=bool)
    indices1 = ((voxels1 - min_bounds) / voxel_size).astype(int)
    grid1[indices1[:, 0], indices1[:, 1], indices1[:, 2]] = True

    # Create an empty 3D grid for the second voxel set
    grid2 = np.zeros(grid_dimensions, dtype=bool)
    indices2 = ((voxels2 - min_bounds) / voxel_size).astype(int)
    grid2[indices2[:, 0], indices2[:, 1], indices2[:, 2]] = True

    # Create an ImageData object for the combined grid
    grid = pv.ImageData()
    grid.dimensions = tuple(np.array(grid_dimensions) + 1)  # Add 1 for point-to-cell mapping
    grid.origin = tuple(min_bounds)
    grid.spacing = (voxel_size, voxel_size, voxel_size)

    # Add both voxel masks as separate cell data
    grid.cell_data["voxels1"] = grid1.flatten(order="F")
    grid.cell_data["voxels2"] = grid2.flatten(order="F")

    # Extract voxels for both sets
    threshold1 = grid.threshold(0.5, scalars="voxels1")
    threshold2 = grid.threshold(0.5, scalars="voxels2")

    # Create a PyVista plotter and visualize both voxel sets
    plotter = pv.Plotter()
    plotter.add_mesh(threshold1, color="blue", show_edges=True, opacity=1, label="Healthy Bone")
    plotter.add_mesh(threshold2, color="yellow", show_edges=True, opacity=1, label="Tumor")
    plotter.add_legend()
    plotter.show_grid()
    plotter.show()

def find_neighboring_voxels(voxels, radius):

    # Randomly select one voxel as the centroid
    centroid = voxels[random.randint(0, len(voxels) - 1)]

    # Calculate the Euclidean distance of each voxel from the centroid
    distances = np.linalg.norm(voxels - centroid, axis=1)

    # Select voxels within the sphere (distance <= radius)
    neighboring_voxels = voxels[distances <= radius]

    return neighboring_voxels

def remove_overlap(voxel1, voxel2):

    # Convert voxel arrays to sets of tuples for easy comparison
    set_voxel1 = set(map(tuple, voxel1))
    set_voxel2 = set(map(tuple, voxel2))

    # Perform set difference to remove overlapping voxels
    unique_voxel1 = np.array(list(set_voxel1 - set_voxel2))

    return unique_voxel1

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

def translate_voxels(voxels, distance, axis='z'):

    # Define axis of vector
    if axis == 'x':
        vector = np.array([1, 0, 0])
    elif axis == 'y':
        vector = np.array([0, 1, 0])
    elif axis == 'z':
        vector = np.array([0, 0, 1])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    # Apply the translation
    translated_voxels = distance * vector + voxels
    return translated_voxels

def create_CP(i_size, j_size, plane_center, angle_y=0, angle_z=0):
    """
    Create a plane and rotate it about its center without moving the center.

    Args:
        i_size: Size of the plane along the first axis.
        j_size: Size of the plane along the second axis.
        plane_center: The center of the plane (3D coordinates).
        angle_y: Rotation angle about the Y-axis in degrees.
        angle_z: Rotation angle about the Z-axis in degrees.

    Returns:
        A PyVista plane rotated about its center.
    """
    # Create the initial plane
    plane = pv.Plane(center=plane_center - np.array([1, 1, 1]),
                     direction=(1, 0, 0),  # Normal direction of the plane
                     i_size=i_size,
                     j_size=j_size)

    # Extract plane points
    plane_points = np.array(plane.points)

    # Shift points to the origin (center of rotation)
    shifted_points = plane_points - plane_center

    # Create the rotation (Intrinsic rotation about Y, then Z)
    rotation = R.from_euler('zy', [angle_z, angle_y], degrees=True)

    # Apply rotation
    rotated_points = rotation.apply(shifted_points)

    # Shift points back to the original center
    rotated_plane_points = rotated_points + plane_center

    # Update the plane's points
    plane.points = rotated_plane_points

    return plane

def visualize_three_voxels_plane(voxels1, voxels2, voxels3, voxel_size, plane):

    # Calculate the grid dimensions based on the combined range of all three voxel sets
    min_bounds = np.minimum.reduce([voxels1.min(axis=0), voxels2.min(axis=0), voxels3.min(axis=0)]) - voxel_size / 2
    max_bounds = np.maximum.reduce([voxels1.max(axis=0), voxels2.max(axis=0), voxels3.max(axis=0)]) + voxel_size / 2
    grid_dimensions = ((max_bounds - min_bounds) // voxel_size + 1).astype(int)

    # Create an empty 3D grid for each voxel set
    grid1 = np.zeros(grid_dimensions, dtype=bool)
    indices1 = ((voxels1 - min_bounds) / voxel_size).astype(int)
    grid1[indices1[:, 0], indices1[:, 1], indices1[:, 2]] = True

    grid2 = np.zeros(grid_dimensions, dtype=bool)
    indices2 = ((voxels2 - min_bounds) / voxel_size).astype(int)
    grid2[indices2[:, 0], indices2[:, 1], indices2[:, 2]] = True

    grid3 = np.zeros(grid_dimensions, dtype=bool)
    indices3 = ((voxels3 - min_bounds) / voxel_size).astype(int)
    grid3[indices3[:, 0], indices3[:, 1], indices3[:, 2]] = True

    # Create an ImageData object for the combined grid
    grid = pv.ImageData()
    grid.dimensions = tuple(np.array(grid_dimensions) + 1)  # Add 1 for point-to-cell mapping
    grid.origin = tuple(min_bounds)
    grid.spacing = (voxel_size, voxel_size, voxel_size)

    # Add voxel masks as separate cell data
    grid.cell_data["voxels1"] = grid1.flatten(order="F")
    grid.cell_data["voxels2"] = grid2.flatten(order="F")
    grid.cell_data["voxels3"] = grid3.flatten(order="F")

    # Extract voxels for all sets
    threshold1 = grid.threshold(0.5, scalars="voxels1")
    threshold2 = grid.threshold(0.5, scalars="voxels2")
    threshold3 = grid.threshold(0.5, scalars="voxels3")

    # Create a PyVista plotter and visualize all voxel sets
    plotter = pv.Plotter()
    plotter.add_mesh(threshold1, color="red", show_edges=True, opacity=1, label="Healthy Bone Loss")
    plotter.add_mesh(threshold2, color="blue", show_edges=True, opacity=1, label="Healthy Bone Remains")
    plotter.add_mesh(threshold3, color="yellow", show_edges=True, opacity=1, label="Tumor")

    plotter.add_mesh(plane, color="red", opacity=0.8, label="Cutting Plane")

    plotter.add_legend()
    plotter.show_grid()
    plotter.show()

def find_identical_rows(*arrays):
    """
    Find identical rows across multiple numpy arrays.

    Args:
        *arrays: A variable number of numpy arrays (each must have the same number of columns).

    Returns:
        numpy.ndarray: An array of identical rows across all input arrays.
    """
    if len(arrays) < 2:
        raise ValueError("At least two arrays are required for comparison.")

    # Ensure all arrays have the same number of columns
    num_columns = arrays[0].shape[1]
    for array in arrays:
        if array.shape[1] != num_columns:
            raise ValueError("All arrays must have the same number of columns.")

    # Use structured arrays to find intersections
    dtype = [('', arrays[0].dtype)] * num_columns
    structured_arrays = [array.view(dtype) for array in arrays]

    # Find the intersection across all arrays
    identical_rows = structured_arrays[0]
    for structured_array in structured_arrays[1:]:
        identical_rows = np.intersect1d(identical_rows, structured_array)

    # Restore the original format
    return identical_rows.view(arrays[0].dtype).reshape(-1, num_columns)

def place_CP(healthy, tumor, angle_y, angle_z, plot_on=False):

    # generate rotation matrix
    rot_y = R.from_euler('y', angle_y, degrees=True)
    rot_z = R.from_euler('z', angle_z, degrees=True)
    rot_combined = rot_z * rot_y
    rot_inverse = rot_combined.inv()

    # Combined rotation for the cylinder (Y then Z)
    rotated_healthy = rot_combined.apply(healthy)
    rotated_tumor = rot_combined.apply(tumor)

    # find the maximum X coordinate of tumor
    CPx = np.min(rotated_tumor[:, 0])
    rotated_center = rotated_tumor[np.argmin(rotated_tumor[:, 0])]

    # segment bone loss and bone remains
    bone_loss = rotated_healthy[rotated_healthy[:, 0] >= CPx]
    bone_remains = rotated_healthy[rotated_healthy[:, 0] < CPx]

    # restore the bone and plane
    restored_bone_loss = rot_inverse.apply(bone_loss)
    restored_bone_remains = rot_inverse.apply(bone_remains)
    restored_center = rot_inverse.apply(rotated_center)

    # create cutting plane
    plane = create_CP(i_size=100, j_size=100, plane_center=restored_center, angle_y=-angle_y, angle_z=-angle_z)

    if plot_on:
        visualize_three_voxels_plane(restored_bone_loss, restored_bone_remains, tumor, voxel_size, plane)

    return restored_bone_loss, restored_bone_remains, plane


def place_new_CP(healthy, tumor, previous_bone_loss, previous_bone_remains, previous_CP, angle_y, angle_z, first_CP=False):

    # place the new CP and compute the new bone_loss/bone_remains
    bone_loss, bone_remains, CP = place_CP(healthy, tumor, angle_y, angle_z)

    # combine bone_loss
    bone_loss = np.round(bone_loss, decimals=4)
    previous_bone_loss = np.round(previous_bone_loss, decimals=4)
    if first_CP:
        previous_bone_loss = bone_loss
    bone_loss_all = find_identical_rows(bone_loss, previous_bone_loss)

    # combine CPs
    CP_all = CP + previous_CP

    # combine bone_loss
    if first_CP:
        previous_bone_remains = bone_remains
    bone_remains_all = np.vstack([previous_bone_remains, bone_remains])
    bone_remains_all = np.unique(bone_remains_all, axis=0)

    return bone_loss_all, bone_remains_all, CP_all


def place_CP_optimization(healthy, tumor, angle_y, angle_z):
    """
    Optimized version of place_CP for use during gradient-based optimization.

    Args:
        healthy (torch.Tensor): Tensor representing the healthy bone points.
        tumor (torch.Tensor): Tensor representing the tumor points.
        angle_y (torch.Tensor): Rotation angle around the Y-axis (requires_grad=True).
        angle_z (torch.Tensor): Rotation angle around the Z-axis (requires_grad=True).

    Returns:
        torch.Tensor: Restored bone loss points after applying rotation and segmentation.
    """
    # Convert angles from degrees to radians
    angle_y_rad = torch.deg2rad(angle_y)
    angle_z_rad = torch.deg2rad(angle_z)

    # Rotation matrix for Y-axis
    rot_y = torch.stack([
        torch.stack([torch.cos(angle_y_rad), torch.zeros_like(angle_y_rad), torch.sin(angle_y_rad)], dim=0),
        torch.stack([torch.zeros_like(angle_y_rad), torch.ones_like(angle_y_rad), torch.zeros_like(angle_y_rad)], dim=0),
        torch.stack([-torch.sin(angle_y_rad), torch.zeros_like(angle_y_rad), torch.cos(angle_y_rad)], dim=0)
    ])

    # Rotation matrix for Z-axis
    rot_z = torch.stack([
        torch.stack([torch.cos(angle_z_rad), -torch.sin(angle_z_rad), torch.zeros_like(angle_z_rad)], dim=0),
        torch.stack([torch.sin(angle_z_rad), torch.cos(angle_z_rad), torch.zeros_like(angle_z_rad)], dim=0),
        torch.stack([torch.zeros_like(angle_z_rad), torch.zeros_like(angle_z_rad), torch.ones_like(angle_z_rad)], dim=0)
    ])

    rot_y = rot_y.squeeze()
    rot_z = rot_z.squeeze()

    # Combined rotation matrix (Z * Y)
    rot_combined = torch.matmul(rot_z, rot_y)  # Matrix multiplication
    rot_inverse = rot_combined.T  # Transpose for inverse

    # Apply rotation to healthy and tumor points
    rotated_healthy = torch.matmul(healthy, rot_combined.T)
    rotated_tumor = torch.matmul(tumor, rot_combined.T)

    # Find the minimum X coordinate of the tumor
    CPx = torch.min(rotated_tumor[:, 0])

    # Differentiable approximation for bone_loss
    bone_loss_mask = torch.sigmoid(100 * (rotated_healthy[:, 0] - CPx))
    bone_loss_count = bone_loss_mask.sum()

    # Return the differentiable loss
    return bone_loss_count







