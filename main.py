import numpy as np
import random
import torch
from utils import place_new_CP, visualize_three_voxels_plane, find_neighboring_voxels, remove_overlap, place_CP_optimization, visualize_voxels, visualize_two_voxels

if __name__ == "__main__":
    random.seed(42)

    # Load voxels
    voxels = np.load("voxels.npy")
    voxel_size = 2.0

    # generate tumor and healthy voxels
    radius = 20
    tumor = find_neighboring_voxels(voxels, radius)
    #np.save("tumor.npy", tumor)
    healthy = remove_overlap(voxels, tumor)

    # print("total number of voxels:")
    # print(np.size(voxels))
    # print("number of voxels in tumor:")
    # print(np.size(tumor))
    # print("number of voxels in healthy bone:")
    # print(np.size(healthy))

    # set torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Selected Device:", device)

    visualize_voxels(voxels, voxel_size)
    visualize_two_voxels(healthy, tumor, voxel_size)

    # initial guess of angle_y and angle_z
    angle_y_t = torch.tensor([0.0], requires_grad=True, dtype=torch.float64)
    angle_z_t = torch.tensor([-90.0], requires_grad=True, dtype=torch.float64)
    healthy_t = torch.tensor(healthy, requires_grad=False, dtype=torch.float64)
    tumor_t = torch.tensor(tumor, requires_grad=False, dtype=torch.float64)

    # define optimizer
    optimizer = torch.optim.SGD([angle_y_t, angle_z_t], lr=0.1)

    # Optimization loop with bounds
    for step in range(200):
        optimizer.zero_grad()
        num_bone_loss = place_CP_optimization(healthy_t, tumor_t, angle_y_t, angle_z_t)
        num_bone_loss.backward()
        optimizer.step()

        # Clamp variables to enforce bounds
        with torch.no_grad():
            angle_y_t.clamp_(-180.0, 180.0)
            angle_z_t.clamp_(-180.0, 180.0)

        if step % 10 == 0:
            print(f"Step {step}: angle_y = {angle_y_t.item():.4f}, angle_z = {angle_z_t.item():.4f}, loss = {num_bone_loss.item():.4f}")


    angle_y = angle_y_t.detach().cpu().item()
    angle_z = angle_z_t.detach().cpu().item()
    angle_y = np.round(angle_y, decimals=4)
    angle_z = np.round(angle_z, decimals=4)
    print(angle_y)
    print(angle_z)

    # place the first CP based on the first optimized angles
    bone_loss, bone_remains, CP = place_new_CP(healthy, tumor, 0, 0, 0, angle_y, angle_z, first_CP=True)

    # visualize the final plot
    visualize_three_voxels_plane(bone_loss, bone_remains, tumor, voxel_size, CP)

    angle_y = 28
    angle_z = -70
    # place the first CP based on the first optimized angles
    bone_loss, bone_remains, CP = place_new_CP(healthy, tumor, bone_loss, bone_remains, CP, angle_y, angle_z)
    visualize_three_voxels_plane(bone_loss, bone_remains, tumor, voxel_size, CP)

    angle_y = -45
    angle_z = -170
    # place the first CP based on the first optimized angles
    bone_loss, bone_remains, CP = place_new_CP(healthy, tumor, bone_loss, bone_remains, CP, angle_y, angle_z)
    visualize_three_voxels_plane(bone_loss, bone_remains, tumor, voxel_size, CP)