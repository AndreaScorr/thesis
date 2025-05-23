import blenderproc as bproc
import argparse
import bpy
from mathutils import Vector
from mathutils import Euler

import yaml

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import os
import trimesh


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_norm_info(mesh):
    bounding_box = mesh.bounding_box.bounds
    diagonal_length = np.linalg.norm(bounding_box[1] - bounding_box[0])

    return diagonal_length

def update_config(config_path, updates):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config.update(updates)

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

def convert_to_nocs(mesh):
    """
    Convert mesh to Normalized Object Coordinate Space (NOCS).
    """

    scaling_factor = get_norm_info(mesh)
    scaled_vertices = mesh.vertices / scaling_factor

    n_vert = len(scaled_vertices)
    colors = []

    for i in range(n_vert):
        # Normalize x, y, z values to the range [-1, 1]
        r = (scaled_vertices[i, 0])
        r = (r + 1) / 2  # Map to [0, 1]

        g = (scaled_vertices[i, 1])
        g = (g + 1) / 2  # Map to [0, 1]
        
        b = (scaled_vertices[i, 2])
        b = (b + 1) / 2  # Map to [0, 1]
        
        # Convert to color values (0-255) and append to the colors array
        colors.append([int(r * 255), int(g * 255), int(b * 255), 255])

    new_mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=mesh.faces,
                               vertex_colors=colors)

    return new_mesh,scaling_factor

bproc.init()


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()

config = load_config(args.config)

obj_path  = "/home/andrea/Desktop/Thesis_project/Models/obj_000002.ply" #config["glb_path"]
name_file = obj_path.split("/")[-1]
name_file = name_file.removesuffix('.glb')
name_file = name_file.removesuffix('.ply')

#obj_path="models/pencil.glb"
mesh = trimesh.load(obj_path, force='mesh')

mesh_nocs,scaling_factor = convert_to_nocs(mesh)
print(scaling_factor)


config_subfolder = obj_path.split("/")
config_subfolder= config_subfolder[1].removesuffix(".glb")
config_subfolder= config_subfolder[1].removesuffix(".ply")

config_path =  "config/"+config_subfolder

current_file_path = os.path.abspath(__file__)            # Percorso completo del file in esecuzione
current_dir = os.path.dirname(current_file_path)         # Solo la cartella dove si trova il file
output_dir = current_dir.removesuffix("/Nocs")
print(f'output dir {output_dir}')

os.makedirs(config_path, exist_ok=True)

scale_path = output_dir+'/config/'+name_file+'_scale.yaml'

yaml_data = {"scaling_factor" : float(scaling_factor)}
with open(scale_path, 'w') as file:
    yaml.dump(yaml_data, file)


# Create a new Blender mesh
blender_mesh = bpy.data.meshes.new(name="TrimeshMesh")

# Extract vertices and faces from the trimesh mesh
vertices = np.array(mesh_nocs.vertices, dtype=np.float32).flatten()
faces = np.array(mesh_nocs.faces, dtype=np.int32).flatten()

blender_mesh.vertices.add(len(mesh_nocs.vertices))
blender_mesh.vertices.foreach_set("co", vertices)

blender_mesh.loops.add(len(faces))
blender_mesh.polygons.add(len(mesh_nocs.faces))

loop_start = np.arange(0, len(faces), 3, dtype=np.int32)
loop_total = np.full(len(mesh_nocs.faces), 3, dtype=np.int32)

blender_mesh.polygons.foreach_set("loop_start", loop_start)
blender_mesh.polygons.foreach_set("loop_total", loop_total)
blender_mesh.loops.foreach_set("vertex_index", faces)

# Set vertex colors if they exist
if mesh_nocs.visual.kind == 'vertex' and mesh_nocs.visual.vertex_colors is not None:
    vertex_colors = mesh_nocs.visual.vertex_colors[:, :3] / 255.0  # Assuming RGBA, use RGB only, and normalize
    color_layer = blender_mesh.vertex_colors.new(name="Col")
    color_data = color_layer.data

    # Assign colors to vertices
    for poly in blender_mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop_vert_idx = blender_mesh.loops[loop_idx].vertex_index
            color_data[loop_idx].color = (*vertex_colors[loop_vert_idx], 1.0)  # Add alpha channel

blender_mesh.update()
#bproc.writer.write_mesh(output_dir="output_nocs", filename="nocs_mesh.obj")

'''
vertices_save = blender_mesh.get_vertices()
faces_save = blender  
'''
# Create a new object with the mesh
blender_object = bpy.data.objects.new(name="TrimeshObject", object_data=blender_mesh)
bpy.context.collection.objects.link(blender_object)



# Ensure the mesh is a single-user copy
blender_object.data = blender_object.data.copy()

# Create a new material
material = bpy.data.materials.new(name="VertexColorMaterial")
material.use_nodes = True
material.blend_method = 'CLIP'  # Set blend method to BLEND or CLIP to avoid darkening
material.shadow_method = 'NONE'  # Disable shadows to ensure true color display
nodes = material.node_tree.nodes
links = material.node_tree.links

# Clear default nodes
for node in nodes:
    nodes.remove(node)

# Create vertex color nodes
output_node = nodes.new(type='ShaderNodeOutputMaterial')
emission_node = nodes.new(type='ShaderNodeEmission')
vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
vertex_color_node.layer_name = "Col"

# Link the nodes
links.new(vertex_color_node.outputs['Color'], emission_node.inputs['Color'])
links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

# Assign the material to the object
if blender_object.data.materials:
    blender_object.data.materials[0] = material
else:
    blender_object.data.materials.append(material)

# Continue with BlenderProc operations
# bproc_mesh = bproc.object.create_from_blender_mesh(blender_mesh, object_name="object_nocs")

# mat = np.identity(4)
# mat[3,:3] = np.array([0.0,0.3,1.0])

# bproc_mesh.set_local2world_mat(mat)
# obj_location = bproc_mesh.get_location()

#blender_object.location += Vector((0.1,0.1,1.))  
#blender_object.location += Vector((0.3,0.8,3.))  

obj_location = np.array(blender_object.location)


# Define how many views to render
num_views = 3# or any number provided by the user

# Function to calculate spherical coordinates
def spherical_coords(num_views, radius=2, theta_range=(0, np.pi / 2)):
    coords = []
    for _ in range(num_views):
        theta = np.random.uniform(*theta_range)  # Polar angle, limit to upper hemisphere
        phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        coords.append((x, y, z))
    return coords

# Calculate camera positions
#camera_positions = spherical_coords(num_views, radius=2, theta_range=(0, np.pi / 2))
camera_positions = spherical_coords(num_views, radius=350, theta_range=(0, np.pi / 2))
#camera_positions = spherical_coords(num_views, radius=1500, theta_range=(0, np.pi / 2))
#print(camera_positions)
can2world_matrix_array =[]
# Render views
for pos in camera_positions:
    cam_location = np.array(obj_location) + np.array(pos)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(obj_location - cam_location)
    # Add homog cam pose based on location and rotation
    cam2world_matrix = bproc.math.build_transformation_mat(cam_location, rotation_matrix)
    can2world_matrix_array.append(cam2world_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

#print(cam2world_matrix)

#for i, matrix in enumerate(can2world_matrix_array):
    #print(f"Matrix {i}:\n{matrix}\n")

bproc.python.renderer.RendererUtility.set_world_background([0, 0, 0])
#bproc.python.renderer.RendererUtility.set_max_amount_of_samples(1)
bproc.python.renderer.RendererUtility.set_noise_threshold(0)
bproc.python.renderer.RendererUtility.set_denoiser(None)
#bproc.python.renderer.RendererUtility.set_light_bounces(1, 0, 0, 1, 0, 8, 0)
bpy.context.scene.cycles.filter_width = 0.0
bproc.renderer.enable_segmentation_output(map_by=["instance"])

# Render the whole pipeline
data = bproc.renderer.render()
#print(len(data["colors"][1]))
# Render and save each view
#output_path = "output_nocs2"
#output_path = "outputs/bottle/nocs"


output_subfolder = obj_path.split("/")[-1]
print(output_subfolder)
output_subfolder= output_subfolder.removesuffix(".glb")
print(output_subfolder)
#print(output_subfolder)

output_path = "nocs/"+output_subfolder


os.makedirs(output_path, exist_ok=True)

# bproc.writer.write_hdf5(output_path, data, append_to_existing_output=False)

for i, pos in enumerate(camera_positions):
    rgb = Image.fromarray(np.uint8(data["colors"][i]))
    #print("###")
    #rgb.show()
    rgb.save(os.path.join(output_path, "{:06d}.png".format(i)))


#print(type(can2world_matrix_array))

camera_poses_path = obj_path.split("/")
camera_poses_path= camera_poses_path[1].removesuffix(".glb")
camera_poses_path =  "config/"+camera_poses_path


#print(camera_poses_path)
cam2Pos_path = output_dir+'/config/'+name_file+'_cam2Pos.yaml'
can2world_matrix_list = [matrix.tolist() for matrix in can2world_matrix_array]
yaml_data_cam = {"cam2Pos" : can2world_matrix_list}
with open(cam2Pos_path, 'w') as file:
    yaml.dump(yaml_data_cam, file,default_flow_style=True)


print(scale_path)
print(cam2Pos_path)

update_config(args.config,
              {"scale_path": scale_path,
               "cam2Pos_path": cam2Pos_path}

              
              )