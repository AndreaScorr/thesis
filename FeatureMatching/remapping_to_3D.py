import json
import cv2

import numpy as np
import matplotlib
import yaml

import open3d as o3d
import open3d.core as o3c
import ImageUtils as IU

import os
def image_to_pointCloud(img):
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]


    r = r.reshape(-1, 1)  # Vettore colonna
    g = g.reshape(-1, 1)  # Vettore colonna
    b = b.reshape(-1, 1)  # Vettore colonna

    #print(r)


    #print(r.shape)  # (262144, 1)
    #print(g.shape)  # (262144, 1)
    #print(b.shape)  # (262144, 1)

    pointCloud = np.concatenate((r, g, b), axis=1)
    #print(pointCloud)



    filtered_points = []

    # Scorriamo ogni riga del pointCloud
    for i in range(pointCloud.shape[0]):
        # Estrai il punto corrente
        point = pointCloud[i]
        
        # Controlla se il punto non è [0, 0, 0]
        if not np.array_equal(point, [0, 0, 0]) and not np.array_equal(point,[1,1,1]):
            # Aggiungilo alla lista dei punti validi
            filtered_points.append(point)

    # Converti la lista in un array numpy
    pointCloud = np.array(filtered_points)

    f = open("points.txt", "w")

    for i in range(pointCloud.shape[0]):
        f.write(str(pointCloud[i])+"\n")
        
    f.close()

    #print(pointCloud.shape)    
    return pointCloud





def nocs_to_mesh(points, scaling_factor,obj_id):

    "convert normalized object coordinate space to  mesh"
    mesh = []
    models_info_path="/home/andrea/Desktop/Thesis_project/Models/models_info.json"
    # Carica le dimensioni reali dell’oggetto
    with open(models_info_path, "r") as f:
        models_info = json.load(f)

    #scaling_factor=IU.compute_model_diameter(None,obj_id,models_info_path=models_info_path)
    
    for i in range(points.shape[0]):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]

        x=x/255.0
        y=y/255.0
        z=z/255.0

        x = x*2-1
        y = y*2-1
        z = z*2-1

        #x = x * size_x
        #y = y * size_y
        #z = z * size_z

        reconstructed_vertex = [x * scaling_factor, y * scaling_factor, z * scaling_factor]

        mesh.append(reconstructed_vertex)

    

    return np.array(mesh)


def features_nocs_to_mesh(points, scaling_factor,obj_id):

    "convert normalized object coordinate space to  mesh"
    mesh = []
    models_info_path="/home/andrea/Desktop/Thesis_project/Models/models_info.json"
    # Carica le dimensioni reali dell’oggetto
    with open(models_info_path, "r") as f:
        models_info = json.load(f)

    #scaling_factor=IU.compute_model_diameter(None,obj_id,models_info_path=models_info_path)

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]

        x=x/255.0
        y=y/255.0
        z=z/255.0

        x = x*2-1
        y = y*2-1
        z = z*2-1

        #x = x * size_x
        #y = y * size_y
        #z = z * size_z
        reconstructed_vertex = [x * scaling_factor, y * scaling_factor, z * scaling_factor]

        mesh.append(reconstructed_vertex)

    

    return np.array(mesh)



def generate_pointCloud(mesh,list_point):

    colors = np.full((len(mesh), 3), [0.0, 0.0, 1.0])  # Blu per tutti i punti
    pcd = o3d.t.geometry.PointCloud(mesh)

    # Assegna i colori alla mesh
    pcd.point["colors"] = o3d.core.Tensor(colors, o3d.core.float32)
    #pcd.point["colors"][5] = [1.0, 0.0, 0.0]  # Imposta un punto specifico a rosso

    # **Definizione corretta dei punti e dei colori**
    '''
    picked_3d_pts = [
        (1.0, 2.0, 3.0),  # Punto 1
        (4.5, 5.2, 6.3),  # Punto 2
         (1.0, 2.0, 3.0),  # Punto 1
        (4.5, 5.2, 6.3),  # Punto 2
         (1.0, 2.0, 3.0),  # Punto 1
        (4.5, 5.2, 6.3)  # Punto 2
    ]
    '''
    picked_3d_pts=list_point
    #print(picked_3d_pts)
    picked_colors = [(1,0,0)]*len(picked_3d_pts)
    
    '''
    picked_colors = [
        (1.0, 0.0, 0.0),  # Rosso per il punto 1
        (1.0, 1, 0.0),  # Verde per il punto 2
        (1.0, 0.0, 0.0),  # Rosso per il punto 1
        (1.0, 1, 0.0),  # Verde per il punto 2
        (1.0, 0.0, 0.0),  # Rosso per il punto 1
        (1.0, 1, 0.0)  # Verde per il punto 2
    ]
    '''
        # **Converti in array NumPy**
    picked_3d_pts = np.array(picked_3d_pts, dtype=np.float64)
    picked_colors = np.array(picked_colors, dtype=np.float64)

    # **Crea il PointCloud per i punti scelti**
    picked_point_cloud = o3d.geometry.PointCloud()
    picked_point_cloud.points = o3d.utility.Vector3dVector(picked_3d_pts)
    picked_point_cloud.colors = o3d.utility.Vector3dVector(picked_colors)

    # **Bounding box per il centro della vista**
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center().cpu().numpy()

    # **Visualizzazione**
    #o3d.visualization.draw_geometries([pcd.to_legacy(), picked_point_cloud],zoom=0.3412,front=[0, 0, -1],lookat=center,up=[0, -1, 0])



def load_mesh_points(path_to_mesh):
    mesh = o3d.io.read_triangle_mesh(path_to_mesh)
    mesh.remove_duplicated_vertices()  # opzionale
    mesh.remove_unreferenced_vertices()  # opzionale
    return np.asarray(mesh.vertices)



'''

#folder_path ="nocs/albero"
#folder_path ="nocs/tennis"
folder_path="nocs/basketball"
config_subfolder = folder_path.split("/")[1]
#print(config_subfolder)
config_path =  "config/"+config_subfolder
scalefactor_path=config_path+"/scale.yaml"

with open(scalefactor_path, "r") as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
scalefactor = data["scaling_factor"]



#folder_path = "output_nocs2"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    print(file_path)
    img = cv2.imread(file_path,cv2.IMREAD_COLOR_RGB)
    pc = image_to_pointCloud(img)
    mesh= nocs_to_mesh(pc,scalefactor)
    generate_pointCloud(mesh)


    
f = open("mesh2.txt", "w")

for i in range(len(mesh)):
    f.write(str(mesh[i])+"\n")
    
f.close()'''