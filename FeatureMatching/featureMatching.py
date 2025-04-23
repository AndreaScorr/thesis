from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os
import cv2
import yaml
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import json
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import remapping_to_3D as r3D
import torch
torch.manual_seed(42)


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


def segment_blender_render():

  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
  args = parser.parse_args()
  config = load_config(args.config)

 



  images_path= config["blender_render_path"]
  #print(images_path)
  model = LangSAM()
  
  assets_amount = 0
  for root_dir, cur_dir, files in os.walk(images_path):
    assets_amount += len(files)

  ##print(assets_amount)
  assets_data_type =".png"
  for i in range(3): #assets_amount
    image_path = os.path.join(images_path, f"{str(i).zfill(6)}{assets_data_type}")
    image_pil = Image.open(image_path).convert("RGB")

    text_prompt = config["text_prompt"]
    image_pil = Image.open(image_path).convert("RGB")
    results = model.predict([image_pil], [text_prompt])

    #extract the mask
    mask = results[0]["masks"]


    # Convert to numpy array
    image_np = np.array(image_pil)  # (H, W, 3)
    mask_np = np.array(mask).astype(np.uint8).squeeze()  # (H, W)
    mask_path= images_path+"/masks"
    update_config(args.config,{"blender_render_mask_path":mask_path})
    mask_file_name= image_path.split("/")[-1]
    os.makedirs(mask_path, exist_ok=True)
    mask_path = os.path.join(mask_path, mask_file_name)
    cv2.imwrite(mask_path, mask_np * 255)



torch.cuda.empty_cache()




class Dinov2Matcher:

  def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=900, half_precision=False, device="cuda"):
    self.repo_name = repo_name
    self.model_name = model_name
    self.smaller_edge_size = smaller_edge_size
    self.half_precision = half_precision
    self.device = device

    if self.half_precision:
      self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
    else:
      self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

    self.model.eval()

    self.transform = transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
      ])

  # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
  def prepare_image(self, rgb_image_numpy):
    image = Image.fromarray(rgb_image_numpy)
    image_tensor = self.transform(image)
    resize_scale = image.width / image_tensor.shape[2]

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size # crop a bit from right and bottom parts
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
    return image_tensor, grid_size, resize_scale
  
  def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
    cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
    image = Image.fromarray(cropped_mask_image_numpy)
    resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
    resized_mask = np.asarray(resized_mask).flatten()
    return resized_mask
  
  def extract_features(self, image_tensor):
    with torch.inference_mode():
      if self.half_precision:
        image_batch = image_tensor.unsqueeze(0).half().to(self.device)
      else:
        image_batch = image_tensor.unsqueeze(0).to(self.device)

      tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
    return tokens.cpu().numpy()
  
  def idx_to_source_position(self, idx, grid_size, resize_scale):
    row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
    col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
    return row, col
  
  def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
    pca = PCA(n_components=3)
    if resized_mask is not None:
      tokens = tokens[resized_mask]
    reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
    if resized_mask is not None:
      tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
      tmp_tokens[resized_mask] = reduced_tokens
      reduced_tokens = tmp_tokens
    reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
    normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
    return normalized_tokens



def feature_matching(image1_path, mask1_path, image2_path, mask2_path):
  # Load image and mask
  image1 = cv2.cvtColor(cv2.imread(image1_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  mask1 = cv2.imread(mask1_path, cv2.IMREAD_COLOR)[:,:,0] > 127

  # Init Dinov2Matcher
  dm = Dinov2Matcher(half_precision=False)

  # Extract features
  image_tensor1, grid_size1, resize_scale1 = dm.prepare_image(image1)
  features1 = dm.extract_features(image_tensor1)


  # Visualization
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,20))
  ax1.imshow(image1)
  resized_mask = dm.prepare_mask(mask1, grid_size1, resize_scale1)
  vis_image = dm.get_embedding_visualization(features1, grid_size1, resized_mask)
  ax2.imshow(vis_image)
  fig.tight_layout()

  # More info
  #print("image1.shape:", image1.shape)
  #print("mask1.shape:", mask1.shape)
  #print("image_tensor1.shape:", image_tensor1.shape)
  #print("grid_size1:", grid_size1)
  #print("resize_scale1:", resize_scale1)

  # Extract image1 features
  #image1 = cv2.cvtColor(cv2.imread(image1_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  #mask1 = cv2.imread(mask1_path, cv2.IMREAD_COLOR)[:,:,0] > 127
  #image_tensor1, grid_size1, resize_scale1 = dm.prepare_image(image1)
  #features1 = dm.extract_features(image_tensor1)

  f = open("features1.txt", "w")
  for feat in features1:
    f.write(str(feat))
  f.close()


  # Extract image2 features
  image2 = cv2.cvtColor(cv2.imread(image2_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  mask2 = cv2.imread(mask2_path, cv2.IMREAD_COLOR)[:,:,0] > 127
  image_tensor2, grid_size2, resize_scale2 = dm.prepare_image(image2)
  features2 = dm.extract_features(image_tensor2)
  f = open("features2.txt", "w")
  for feat in features2:
    f.write(str(feat))
  f.close()
  #print(f'shape image1: {image1.shape}')

  #print(f'shape image2: {image2.shape}')
  # Build knn using features from image1, and query all features from image2
  knn = NearestNeighbors(n_neighbors=1,algorithm='auto', metric='euclidean')
  knn.fit(features1)
  distances, match2to1 = knn.kneighbors(features2)
  match2to1 = np.array(match2to1)
  #print(f'features:{len(match2to1)}')
  plt.plot(sorted(distances.flatten()))
  #plt.show()

  #print("match 2to1",len(match2to1))


  #fig = plt.figure(figsize=(20,10))
  #ax1 = fig.add_subplot(121)
  #ax2 = fig.add_subplot(122)

  #ax1.imshow(image1)
  #ax2.imshow(image2)

  # Lista per raccogliere i punti di corrispondenza tra le due immagini
  points1 = []
  points2 = []

  for idx2, (dist, idx1) in enumerate(zip(distances, match2to1)):
    
    row, col = dm.idx_to_source_position(idx1, grid_size1, resize_scale1)
    row = row.item()
    col = col.item()
    xyA = (col, row)
    
    if not mask1[int(row), int(col)].item(): continue # skip if feature is not on the object

    row, col = dm.idx_to_source_position(idx2, grid_size2, resize_scale2)
    xyB = (col, row)
    if not mask2[int(row), int(col)].item(): continue # skip if feature is not on the object

    if np.random.rand() > 0.05: continue # sparsely draw so that we can see the lines...

    con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color=np.random.rand(3,))
    #ax2.add_artist(con)
    # Aggiungiamo i punti di corrispondenza alle liste
    #print("xyA",xyA)
    points1.append(xyA)
    points2.append(xyB)

  
#plt.show()

  #print("points1 shape:",points1[0])
  #print(len(points2))

  # Convertiamo le liste in array numpy per RANSAC
  points1 = np.array(points1)
  points2 = np.array(points2)

  # Imposta il seed per la riproducibilità
  np.random.seed(42)
  random.seed(42)
  cv2.setRNGSeed(42)  # Solo se vuoi forzare anche il generatore interno di OpenCV
  # Applichiamo RANSAC per trovare l'omografia
  H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransacReprojThreshold=9.0)

  # La maschera contiene 1 per gli inliers e 0 per gli outliers
  inliers = mask.ravel() == 1

  # Filtriamo i punti di corrispondenza validi (inliers)
  valid_points1 = points1[inliers]
  valid_points2 = points2[inliers]

  # Creiamo il grafico per visualizzare le corrispondenze valide
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

  ax1.imshow(image1)
  ax2.imshow(image2)

  # Disegniamo le corrispondenze valide
  for xyA, xyB in zip(valid_points1, valid_points2):
      con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color=np.random.rand(3,))
      ax2.add_artist(con)

  #plt.show()
  
  return points1,points2,image1.shape,image2.shape,H

def find_nearest_non_black_white(img, x, y, max_search_radius=10):
    """
    Cerca il pixel più vicino a (x, y) che non sia [0,0,0] (nero) o [1,1,1] (bianco).
    max_search_radius: Distanza massima di ricerca intorno al punto.
    """
    h, w, _ = img.shape  # Dimensioni immagine

    for r in range(1, max_search_radius + 1):  # Espandi il raggio di ricerca
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = x + dx, y + dy  # Nuove coordinate

                # Controlla se siamo dentro i limiti dell'immagine
                if 0 <= nx < w and 0 <= ny < h:
                    pixel = img[ny, nx]  # Nota: (y, x) in OpenCV
                    if not np.array_equal(pixel, [0, 0, 0]) and not np.array_equal(pixel, [1, 1, 1]):
                        return nx, ny, pixel  # Restituisce la posizione e il colore valido

    return x, y, img[y, x]  # Se non trova nulla, restituisce lo stesso punto

segment_blender_render()


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()
config = load_config(args.config)
image1_path = config["image_path"] #real_image
text_prompt= config["text_prompt"]
mask1_path = config["mask_path"]+"/"+text_prompt+"_"+image1_path.split("/")[-1]
##print(mask1_path)


assets_data_type= ".png"

images2_path = config["blender_render_path"]
#for i in range(2):
#  image2_path = os.path.join(images2_path, f"{str(i).zfill(6)}{assets_data_type}")

masks_path= config["blender_render_mask_path"]
#for i in range(1):
#  mask2_path = os.path.join(masks_path, f"{str(i).zfill(6)}{assets_data_type}")


##print(mask1_path)
#img1_pixel_matches,img2_pixel_matches,shape1,shape2=feature_matching(image1_path,mask1_path,image2_path,mask2_path)

i=1
image2_path = os.path.join(images2_path, f"{str(i).zfill(6)}{assets_data_type}")
#print(f'image2_path : {image2_path}')
mask2_path = os.path.join(masks_path, f"{str(i).zfill(6)}{assets_data_type}")
img1_pixel_matches,img2_pixel_matches,shape1,shape2,H=feature_matching(image1_path,mask1_path,image2_path,mask2_path)

'''
for i in range(3):
  image2_path = os.path.join(images2_path, f"{str(i).zfill(6)}{assets_data_type}")
  #print(f'image2_path : {image2_path}')
  mask2_path = os.path.join(masks_path, f"{str(i).zfill(6)}{assets_data_type}")
  img1_pixel_matches,img2_pixel_matches,shape1,shape2,H=feature_matching(image1_path,mask1_path,image2_path,mask2_path)
  if(len(img1_pixel_matches)>6):
     break'''
      
subfolder_nocs=image1_path.split("/")[-1].removesuffix(".jpg")
nocs_path = "/home/andrea/Desktop/Thesis_project/nocs/obj_000014.ply" #config["mask_path"].replace("/Segmented/mask","/nocs")+"/"+text_prompt+"_"+subfolder_nocs+"_sample"
  
image_nocs_path = os.path.join(nocs_path, f"{str(i).zfill(6)}{assets_data_type}")
#print(f'image_nocs path: {image_nocs_path}')
img_nocs =cv2.imread(image_nocs_path,cv2.IMREAD_COLOR_RGB)
plt.figure(0)
  
#print(img2_pixel_matches)
features3Dpoint =[]
for x,y in img2_pixel_matches:
    x1, y1 = int(x), int(y)  # Convert to int
    ##print(x)
    #cv2.circle(img_nocs,(x1,y1), 1, (0, 255, 0), 2)  # (x, y), raggio, colore (BGR), spessore
    if np.all(img_nocs[y1, x1] == [1, 1, 1]) or np.all(img_nocs[y1, x1] == [0, 0, 0]):
        x1,y1,new_color = find_nearest_non_black_white(img_nocs,x1,y1,3)
    else:
        new_color = img_nocs[y1, x1]
    #features3Dpoint.append(img_nocs[y1,x1])
    features3Dpoint.append(tuple(map(int, new_color))) #use the color of the match pixel

    ##print(f'posizione {x1,y1} colore: {img_nocs[y1,x1]}')
    img_nocs[y1,x1]=(0,255,0) #paint the pixel of the match

plt.imshow(img_nocs)
#print(image_nocs_path)
#print()



scale_path= config["scale_path"]
with open(scale_path, 'r') as f:
      data = yaml.safe_load(f)
scaling_factor=data["scaling_factor"]
#pc = r3D.image_to_pointCloud(img_nocs)
#mesh= r3D.nocs_to_mesh(pc,scaling_factor)
#r3D.generate_pointCloud(mesh)

pc = r3D.image_to_pointCloud(img_nocs)
mesh= r3D.nocs_to_mesh(pc,scaling_factor)

features_nocs_to_mesh =r3D.features_nocs_to_mesh(features3Dpoint,scaling_factor)
feature_point_tuple = [tuple(l) for l in features_nocs_to_mesh]
plt.show()

r3D.generate_pointCloud(mesh,feature_point_tuple)




cam_k=[1066.778, 0.0, 312.9869079589844,
      0.0, 1067.487, 241.3108977675438,
      0.0, 0.0, 1.0]

#print(features3Dpoint)


##scale images ##
scale_x = shape2[1] / shape1[1]
scale_y = shape2[0] / shape1[0]


# Converti img_nocs da [0,255] a [0,1] per ottenere coordinate normalizzate
features_nocs_to_mesh = np.array(feature_point_tuple, dtype=np.float32) 

#print(f'image nocs: {img_nocs}')



object_points_3D = np.array(features_nocs_to_mesh, dtype=np.float32)
# Converti i punti 2D in NumPy array
image_points_2D = np.array(img1_pixel_matches, dtype=np.float32)

print(H)


ones = np.ones((image_points_2D.shape[0], 1))

'''
print(ones[0])
points_hom = np.hstack([image_points_2D, ones])  # shape (N, 3)
print(points_hom[0])
# Assicurati che H sia numpy e abbia shape (3, 3)
transformed_hom = (H.T @ points_hom.T).T  # shape (N, 3)
print(transformed_hom[0])

image_points_2D = transformed_hom[:, :2] / transformed_hom[:, 2][:, np.newaxis]  # shape (N, 2)
'''

camera_matrix=np.array([[1066.778, 0.0,      312.9869079589844],
                        [0.0,      1067.487, 241.3108977675438],
                        [0.0,      0.0,      1.0]])

# Nessuna distorsione se l'immagine è già calibrata
dist_coeffs = np.ones((5, 1)) *0

# Usa SolvePnP per ottenere Rotazione e Traslazione

#print(f'shape object points{object_points_3D.shape}')
#print(f'shape image points{image_points_2D.shape}')
#print(camera_matrix.shape)


#retval, rvec, tvec, inliers= cv2.solvePnPRansac(object_points_3D, image_points_2D, camera_matrix, dist_coeffs,iterationsCount=200)
success,rvec,tvec=cv2.solvePnP(object_points_3D, image_points_2D, camera_matrix, dist_coeffs)
print(success)
R, _ = cv2.Rodrigues(rvec)
#print(rvec)
print(f'matrix: {R}')
#print(f'matrix: {R*camera_matrix}')




print("t:",tvec)
'''
rvec=np.array([[-0.988410234451294, -0.012824120000004768, -0.1512639820575714],
  [0.13328857719898224, 0.4036089777946472, -0.9051706790924072],
  [0.07265951484441757, -0.9148417115211487, -0.3972219228744507]])
tvec = np.array([94.04884338378906, -50.48897933959961, 572.3798828125])
'''


'''
"cam_R_m2c": 
[0.07044275850057602, 0.41998282074928284, -0.9047940373420715, 
-0.9961117506027222, -0.01849273405969143, -0.08613616973161697, 
-0.05290782079100609, 0.9073436260223389, 0.4170471429824829],
 "cam_t_m2c": [117.49842071533203, 354.2092590332031, 806.51806640625], "obj_id": 15}
'''

'''
"cam_R_m2c": [-0.988410234451294, -0.012824120000004768, -0.1512639820575714, 
              0.13328857719898224, 0.4036089777946472, -0.9051706790924072, 
              0.07265951484441757, -0.9148417115211487, -0.3972219228744507], 
"cam_t_m2c": [94.04884338378906, -50.48897933959961, 572.3798828125], 
"obj_id": 14

'''



#ground truth rotation matrix mug
R = np.array([
    [-0.98841023, -0.01282412, -0.15126398],
    [ 0.13328858,  0.40360898, -0.90517068],
    [ 0.07265951, -0.91484171, -0.39722192]
])
#tvec = np.array([94.04884338, -50.48897934, 572.37988281])


#ground truth banana
R = np.array([[-0.00015016013639979064, 0.4626128077507019, 0.8865604996681213],
              [0.622158944606781, 0.6941233277320862, -0.36209261417388916], 
              [-0.782891035079956, 0.5515271425247192, -0.2879229187965393]
]) 
#tvec = np.array([-112.10218811035156, 16.280380249023438, 627.1389770507812])



'''
R = np.array([
    [-0.86606628, -0.45950323,  0.19694093],
    [ 0.07293312, -0.50585407, -0.85953027],
    [ 0.49458033, -0.73004675,  0.47161633]
], dtype=np.float32)
'''
#tvec = np.array([-18.96696091, -60.08344269, 825.09411621], dtype=np.float32)

#rvec, _ = cv2.Rodrigues(R)


# Bounding box 3D nell'object frame (NOCS normalizzato in [0,1])
bbox_3D = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5]
], dtype=np.float32)



#take the size from the groud truth (this is a parameter that is missing)
with open("/home/andrea/Desktop/Thesis_project/Models/models_info.json", "r") as f:
    models_info = json.load(f)

obj_id = "14"  # oppure "2", ecc.
size_x = models_info[obj_id]["size_x"]
size_y = models_info[obj_id]["size_y"]
size_z = models_info[obj_id]["size_z"]

scaling_factor = np.array([size_x, size_y, size_z], dtype=np.float32)

# Scala la bounding box con lo scaling dell’oggetto reale
bbox_3D_scaled = bbox_3D * scaling_factor


# Proietta i punti 3D nel piano immagine
projected_points, _ = cv2.projectPoints(bbox_3D_scaled, rvec, tvec, camera_matrix, dist_coeffs)
projected_points = projected_points.reshape(-1, 2)  # (8, 2)


image_with_bbox = cv2.cvtColor(cv2.imread(image1_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#projected_points = projected_points.astype(int)
projected_points = np.where(np.isnan(projected_points), -1, projected_points).astype(int)

# Definisci gli edge (collegamenti) della box
edges = [
    (0,1), (1,2), (2,3), (3,0),  # base
    (4,5), (5,6), (6,7), (7,4),  # top
    (0,4), (1,5), (2,6), (3,7)   # vertical edges
]

for start, end in edges:
  pt1_raw = projected_points[start]
  pt2_raw = projected_points[end]

  # Check per NaN o inf
  if (
      pt1_raw is None or pt2_raw is None or
      np.any(np.isnan(pt1_raw[:2])) or
      np.any(np.isnan(pt2_raw[:2])) or
      np.any(np.isinf(pt1_raw[:2])) or
      np.any(np.isinf(pt2_raw[:2]))
  ):
      #print(f"Skipping edge {start}-{end} due to invalid point(s)")
      continue

  # Converti in tuple di int
  try:
      pt1 = tuple(int(round(float(x))) for x in pt1_raw[:2])
      pt2 = tuple(int(round(float(x))) for x in pt2_raw[:2])
  except Exception as e:
      #print(f"Error converting points {pt1_raw} or {pt2_raw}: {e}")
      continue

  #print(" ###")
  #print(f"pt1: {pt1} (types: {[type(x) for x in pt1]})")
  #print(f"pt2: {pt2} (types: {[type(x) for x in pt2]})")
  #print(" ###")

  cv2.line(image_with_bbox, pt1, pt2, color=(0,255,0), thickness=2)

# Mostra l’immagine
plt.figure(figsize=(10, 8))
plt.imshow(image_with_bbox)
plt.title("3D Bounding Box Projection")
plt.axis("off")
plt.show()