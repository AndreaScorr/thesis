import open3d as o3d

# Carica le mesh
mesh_source = o3d.io.read_triangle_mesh("/home/andrea/Desktop/Thesis_project/Models/TRELLIS/000014/sample.glb")
mesh_target = o3d.io.read_triangle_mesh("Models/obj_000014.ply")
print(mesh_source)
# Campiona punti
pcd_source = mesh_source.sample_points_uniformly(number_of_points=100000)
pcd_target = mesh_target.sample_points_uniformly(number_of_points=100000)

# Allinea con ICP
threshold = 0.01  # distanza massima per considerare una corrispondenza
trans_init = o3d.geometry.Transformation.identity()

reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_source, pcd_target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Trasforma la mesh source secondo la trasformazione trovata
mesh_source.transform(reg_p2p.transformation)

# Visualizza l'allineamento
o3d.visualization.draw_geometries([pcd_source.paint_uniform_color([1, 0, 0]), 
                                   pcd_target.paint_uniform_color([0, 1, 0])])
