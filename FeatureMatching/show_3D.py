import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # serve per abilitare il 3D in matplotlib

def load_nocs_from_npy(npy_path):
    nocs = np.load(npy_path)
    if nocs.ndim == 3 and nocs.shape[2] == 3:  # (H, W, 3)
        nocs = nocs.reshape(-1, 3)
    elif nocs.ndim != 2 or nocs.shape[1] != 3:
        raise ValueError("Il file .npy deve contenere una matrice di forma (N, 3) o (H, W, 3)")
    
    # Filtra [0,0,0] e [1,1,1] (o [255,255,255] se in int)
    return np.array([pt for pt in nocs if not np.allclose(pt, 0) and not np.allclose(pt, 1) and not np.allclose(pt, 255)])

def normalize_and_scale(nocs_points, scaling_factor):
    # Se i valori sono in [0, 255], normalizza
    if nocs_points.max() > 1.5:
        nocs_points = nocs_points / 255.0

    # Porta da [0,1] a [-1,1]
    nocs_normalized = nocs_points * 2 - 1

    # Scala rispetto alle dimensioni reali
    return nocs_normalized * scaling_factor

def visualize_pointcloud(points, highlight_points=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.full_like(points, [0.0, 0.0, 1.0]))  # blu

    geometries = [pcd]

    if highlight_points is not None:
        hp = o3d.geometry.PointCloud()
        hp.points = o3d.utility.Vector3dVector(np.array(highlight_points))
        hp.colors = o3d.utility.Vector3dVector(np.full((len(highlight_points), 3), [1.0, 0.0, 0.0]))  # rosso
        geometries.append(hp)

    o3d.visualization.draw_geometries(geometries)

    
def plot_with_matplotlib(points):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot dei punti
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1)

    # Imposta etichette assi
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Imposta limiti automatici (o fissi se vuoi)
    ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])

    # Griglia
    ax.grid(True)

    plt.title("Visualizzazione 3D con matplotlib")
    plt.show()

def main():
    # ✅ CONFIGURAZIONE
    npy_path = "/home/andrea/Desktop/ZS6/ZS6D/templates/ycbv_desc/obj_14/000098_uv.npy"
    models_info_path = "/home/andrea/Desktop/Thesis_project/Models/models_info.json"
    obj_id = 1

    # 🔁 Carica scaling factor dal JSON
    with open(models_info_path, 'r') as f:
        models_info = json.load(f)
    scaling_factor = models_info[str(obj_id)]["diameter"] / 10.0  # oppure altro fattore

    # 🔄 Carica e trasforma i punti
    nocs_points = load_nocs_from_npy(npy_path)
    mesh_points = normalize_and_scale(nocs_points, scaling_factor)

    # 👁️ Visualizza
    visualize_pointcloud(mesh_points)
    plot_with_matplotlib(mesh_points)


if __name__ == "__main__":
    main()
