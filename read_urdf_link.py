import trimesh
import os

def generate_collision_box_from_mesh(stl_path):
    mesh = trimesh.load(stl_path)
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    size = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2

    print(f"STL: {os.path.basename(stl_path)}")
    print(f"  Size: {size}")
    print(f"  Center offset: {center}")

    urdf_collision = f"""
  <collision>
    <origin xyz="{center[0]:.5f} {center[1]:.5f} {center[2]:.5f}" rpy="0 0 0"/>
    <geometry>
      <box size="{size[0]:.5f} {size[1]:.5f} {size[2]:.5f}"/>
    </geometry>
  </collision>
"""
    print(urdf_collision)

# 使用示例
generate_collision_box_from_mesh("resources/tita/meshes/base_link.STL")
generate_collision_box_from_mesh("resources/tita/meshes/left_leg_1.STL")
generate_collision_box_from_mesh("resources/tita/meshes/left_leg_2.STL")
generate_collision_box_from_mesh("resources/tita/meshes/left_leg_3.STL")
generate_collision_box_from_mesh("resources/tita/meshes/left_leg_4.STL")
generate_collision_box_from_mesh("resources/tita/meshes/right_leg_1.STL")
generate_collision_box_from_mesh("resources/tita/meshes/right_leg_2.STL")
generate_collision_box_from_mesh("resources/tita/meshes/right_leg_3.STL")
generate_collision_box_from_mesh("resources/tita/meshes/right_leg_4.STL")
