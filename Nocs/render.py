import blenderproc as bproc
import bpy
from mathutils import Vector
import yaml
from mathutils import Euler

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import os
import trimesh
import argparse


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config_path, updates):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config.update(updates)

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

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


bproc.init()



#obj_path = "/media/paolo/Datasets/HouseCat6D/obj_models_small_size_final/cup/cup-green_handle.obj"
#obj_path = "uploads_files_3776671_Drinks.obj"
#obj_path = "models/bottle.obj"

##### insert model path here ! ###
#"models/tennis.glb"
#"models/basketball.glb"
#"models/basketball_texture.glb"
#"models/bottle.glb"

#obj_path="models/pencil.glb"
#obj_path="models/banana_000019_sample.glb"

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()

config = load_config(args.config)

obj_path  = "/home/andrea/Desktop/Thesis_project/Models/obj_000015.ply" #config["glb_path"]

#load the object
objs = bproc.loader.load_obj(obj_path)
obj =objs[0]

light = bproc.types.Light()
light.set_type("POINT")  # Tipo di luce, in questo caso una luce puntiforme
light.set_location([2, -2, 15])
light.set_energy(200)


camera_pose = np.array([
    [1, 0, 0, 2],  # X
    [0, 1, 0, 3],  # Y
    [0, 0, 1, 4],  # Z
    [0, 0, 0, 1]  

]) 

#obj.set_location(obj.get_location() + Vector((0.3,0.8,3.))) 

#new_rotation = Euler((np.radians(-90), 0 , 0), 'XYZ')  # Rotation around x axis -90

#apply rotation

#obj.set_rotation_euler(new_rotation)


#get the path for the cam2Pos config file
'''
config_subfolder = obj_path.split("/")
config_subfolder= config_subfolder[1].removesuffix(".glb")
config_path =  "config/"+config_subfolder
'''

cam2Pos_path=config["cam2Pos_path"]

#open file yaml with cameraPose configuration
with open(cam2Pos_path, "r") as file:
    data = yaml.safe_load(file)  # Carica il contenuto del file


# convert list into numPy arrary
can2world_matrix_array = [np.array(matrix) for matrix in data["cam2Pos"]]
# print transformation matrix
for i, matrix in enumerate(can2world_matrix_array):
    bproc.camera.add_camera_pose(matrix)
    print(f"Matrix {i}:\n{matrix}\n")


bproc.python.renderer.RendererUtility.set_world_background([0.5, 0.5, 0.5])

#render of the scene
data = bproc.renderer.render()
cam_K =         np.array([[1066.778, 0.0,      312.9869079589844],
                          [0.0,      1067.487, 241.3108977675438],
                          [0.0,      0.0,      1.0]])
#bproc.camera.set_intrinsics_from_K_matrix(cam_K, 512, 512)
#get the path for the output folder
output_subfolder = obj_path.split("/")[-1]
print(output_subfolder)
output_subfolder= output_subfolder.removesuffix(".glb")
output_path = "blender_render/"+output_subfolder

# create the folder
os.makedirs(output_path, exist_ok=True)

# Save the images
for i, pos in enumerate(can2world_matrix_array):
    rgb = Image.fromarray(np.uint8(data["colors"][i]))
    print("###")
    
    rgb.save(os.path.join(output_path, "{:06d}.png".format(i)))
    print(os.path.join(output_path, "{:06d}.png".format(i)))

rgb_image = Image.fromarray(np.uint8(data["colors"][0]))
#rgb_image.save(os.path.join(output_path, "rendered_image.png"))


current_file_path = os.path.abspath(__file__)            # Percorso completo del file in esecuzione
current_dir = os.path.dirname(current_file_path)         # Solo la cartella dove si trova il file
current_dir = current_dir.removesuffix("/Nocs")

output_path= current_dir+'/'+output_path
print(f"Render completed and saved {output_path}")
update_config(args.config, 
 {"blender_render_path": output_path}             
)
