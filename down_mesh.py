import pymeshlab

# Load the STL file
ms = pymeshlab.MeshSet()
ms.load_new_mesh('femur.stl')

# Simplify the mesh using the appropriate filter
# Target the desired number of faces (e.g., 50% reduction in triangles)
ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=5000, preserveboundary=True)

# Save the simplified STL file
ms.save_current_mesh('output_simplified.stl')

print("Simplification complete! Saved as 'output_simplified.stl'")

