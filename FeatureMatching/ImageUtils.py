







import cv2
import numpy as np
import torch
import argparse
import json
import os
import torch
from pathlib import Path
from extractor import ViTExtractor
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
import cv2
import yaml
import remapping_to_3D as r3D
from lang_sam import LangSAM
import ImageUtils as img_utils
from math import acos, degrees
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


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


def make_quadratic_crop(image, bbox):
    # Define the bounding box
    x_left, y_top, width, height = bbox

    # Calculate the size of the square crop based on the longer side
    longer_side = max(width, height)
    crop_size = (longer_side, longer_side)

    # Calculate the center of the bounding box
    center_x = x_left + width / 2
    center_y = y_top + height / 2
    crop_size = min(longer_side, int(max(width/2, height/2) * 2))

    # Calculate the coordinates of the top-left corner of the square crop
    crop_x = int(center_x - crop_size / 2)
    crop_y = int(center_y - crop_size / 2)
    
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Check if the crop goes beyond the image boundaries
    if crop_x < 0 or crop_y < 0 or crop_x + crop_size > image.shape[1] or crop_y + crop_size > image.shape[0]:

        # If the crop goes beyond the image boundaries, crop first and add a border using cv2.copyMakeBorder to make the crop quadratic
        crop = image[max(crop_y, 0):min(crop_y+crop_size, image.shape[0]), max(crop_x, 0):min(crop_x+crop_size, image.shape[1])]
        border_size = max(crop_size - crop.shape[1], crop_size - crop.shape[0])
        border_size = max(0, border_size)  # Make sure the border size is not negative
        
        
        if crop_x < 0 or crop_x + crop_size > image.shape[1]:
            left = border_size // 2
            right = border_size - left
            crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_REPLICATE)
        elif crop_y < 0 or crop_y + crop_size > image.shape[0]:
            top = border_size // 2
            bottom = border_size - top
            crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
        else:
            print("Something went wrong during rectifying crop")
            return None

    else:
        # If the crop is within the image boundaries, just crop the image
        crop = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    
    return crop, crop_y, crop_x


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)



def find_correspondences_bop(image_path1, image_path2, extractor, num_pairs: int = 10, load_size: int = 224, layer: int = 9,#9,
                             facet: str = 'key', bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                             stride: int = 4) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
                                                                              Image.Image, Image.Image]:
    """
    finding point correspondences between two images.
    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """
    # extracting descriptors for each image
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess(image_path2, load_size)
    descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size

    # extracting saliency maps for each image
    saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(device))[0]
    saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(device))[0]
    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

    bbs_mask = nn_2[nn_1] == image_idxs

    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device)
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i

    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
        bb_indices_to_show]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=device)[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]
    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
    points1, points2 = [], []
    for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
        x1_show = (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y1_show = (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        x2_show = (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y2_show = (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))
    return points1, points2, image1_pil, image2_pil



def draw_projected_3d_bbox(image, obj_id, rvec, tvec, camera_matrix, dist_coeffs, models_info_path):
    # Definisci la bounding box normalizzata [0, 1] nel frame dell’oggetto
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

    # Carica le dimensioni reali dell’oggetto
    with open(models_info_path, "r") as f:
        models_info = json.load(f)

    size_x = models_info[obj_id]["size_x"]
    size_y = models_info[obj_id]["size_y"]
    size_z = models_info[obj_id]["size_z"]
    scaling_factor = np.array([size_x, size_y, size_z], dtype=np.float32)

    #tvec = np.squeeze(tvec)*scaling_factor
    # Scala la bounding box
    bbox_3D_scaled = bbox_3D * scaling_factor

    # Proiezione nel piano immagine
    projected_points, _ = cv2.projectPoints(bbox_3D_scaled, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    projected_points = np.where(np.isnan(projected_points), -1, projected_points).astype(int)

    image = np.array(image)

    # Carica immagine
   # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Edge della bounding box
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # base
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # vertical edges
    ]

    # Disegna le linee
    for start, end in edges:
        pt1_raw = projected_points[start]
        pt2_raw = projected_points[end]

        if (
            pt1_raw is None or pt2_raw is None or
            np.any(np.isnan(pt1_raw[:2])) or
            np.any(np.isnan(pt2_raw[:2])) or
            np.any(np.isinf(pt1_raw[:2])) or
            np.any(np.isinf(pt2_raw[:2]))
        ):
            continue

        try:
            pt1 = tuple(int(round(float(x))) for x in pt1_raw[:2])
            pt2 = tuple(int(round(float(x))) for x in pt2_raw[:2])
        except Exception:
            continue

        cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)

    # Mostra il risultato
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"3D Bounding Box Projection - Object {obj_id}")
    plt.axis("off")
    plt.show()

def draw_projected_3d_bbox_gt(image, obj_id, rvec, tvec, rvec_gt, tvec_gt, camera_matrix, dist_coeffs, models_info_path):
    
   # Definisci la bounding box normalizzata [0, 1] nel frame dell’oggetto
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

    # Carica le dimensioni reali dell’oggetto
    with open(models_info_path, "r") as f:
        models_info = json.load(f)

    size_x = models_info[obj_id]["size_x"]
    size_y = models_info[obj_id]["size_y"]
    size_z = models_info[obj_id]["size_z"]
    scaling_factor = np.array([size_x, size_y, size_z], dtype=np.float32)

    #tvec = np.squeeze(tvec)*scaling_factor
    # Scala la bounding box
    bbox_3D_scaled = bbox_3D * scaling_factor

    # Proiezione nel piano immagine
    projected_points, _ = cv2.projectPoints(bbox_3D_scaled, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    projected_points = np.where(np.isnan(projected_points), -1, projected_points).astype(int)

    image = np.array(image)

    # Carica immagine
   # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Edge della bounding box
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # base
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # vertical edges
    ]

    # Disegna le linee
    for start, end in edges:
        pt1_raw = projected_points[start]
        pt2_raw = projected_points[end]

        if (
            pt1_raw is None or pt2_raw is None or
            np.any(np.isnan(pt1_raw[:2])) or
            np.any(np.isnan(pt2_raw[:2])) or
            np.any(np.isinf(pt1_raw[:2])) or
            np.any(np.isinf(pt2_raw[:2]))
        ):
            continue

        try:
            pt1 = tuple(int(round(float(x))) for x in pt1_raw[:2])
            pt2 = tuple(int(round(float(x))) for x in pt2_raw[:2])
        except Exception:
            continue

        cv2.line(image, pt1, pt2, color=(255, 255, 0), thickness=2)
    
    
    # Proiezione nel piano immagine
    projected_points2, _ = cv2.projectPoints(bbox_3D_scaled, rvec_gt, tvec_gt, camera_matrix, dist_coeffs)
    projected_points2 = projected_points2.reshape(-1, 2)
    projected_points2 = np.where(np.isnan(projected_points2), -1, projected_points2).astype(int)

    #image = np.array(image)

    # Carica immagine
   # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Edge della bounding box
    edges2 = [
        (0,1), (1,2), (2,3), (3,0),  # base
        (4,5), (5,6), (6,7), (7,4),  # top
        (0,4), (1,5), (2,6), (3,7)   # vertical edges
    ]

    # Disegna le linee
    for start, end in edges2:
        pt1_raw = projected_points2[start]
        pt2_raw = projected_points2[end]

        if (
            pt1_raw is None or pt2_raw is None or
            np.any(np.isnan(pt1_raw[:2])) or
            np.any(np.isnan(pt2_raw[:2])) or
            np.any(np.isinf(pt1_raw[:2])) or
            np.any(np.isinf(pt2_raw[:2]))
        ):
            continue

        try:
            pt1 = tuple(int(round(float(x))) for x in pt1_raw[:2])
            pt2 = tuple(int(round(float(x))) for x in pt2_raw[:2])
        except Exception:
            continue

        cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)

    # Mostra il risultato
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"3D Bounding Box Projection - Object {obj_id}")
    plt.axis("off")
    plt.show()


def input_resize(image, target_size, intrinsics):


    # image: [y, x, c] expected row major
    # target_size: [y, x] expected row major
    # instrinsics: [fx, fy, cx, cy]

    intrinsics = np.asarray(intrinsics)
    x_size, y_size = image.size

    if (y_size / x_size) < (target_size[0] / target_size[1]):
        resize_scale = target_size[0] / y_size
        crop = int((x_size - (target_size[1] / resize_scale)) * 0.5)
        #image = image[:, crop:(x_size-crop), :]
        #image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale
    else:
        resize_scale = target_size[1] / x_size
        crop = int((y_size - (target_size[0] / resize_scale)) * 0.5)
        #image = image[crop:(y_size-crop), :, :]
        #image = cv2.resize(image, (int(target_size[1]), int(target_size[0])))
        intrinsics = intrinsics * resize_scale

    return image, intrinsics

def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def add(pts,R_gt, T_gt, R_est, T_est,models_info_path,obj_id):
    with open(models_info_path, "r") as f:
        models_info = json.load(f)
    center_dist = np.linalg.norm(T_est - T_gt)
    spheres_overlap = center_dist < models_info[str(obj_id)]["diameter"]


    pts_est = transform_pts_Rt(pts, R_est, T_est)
    pts_gt = transform_pts_Rt(pts, R_gt, T_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def compute_model_diameter(model_points):
    """
    Calcola il diametro del modello, cioè la massima distanza tra tutte le coppie di punti.

    Parametri:
    - model_points: array (N, 3) dei punti del modello

    Ritorna:
    - diametro (float)
    """
    distances = cdist(model_points, model_points)  # (N, N)
    return np.max(distances)
'''
def compute_add(R_gt, T_gt, R_est, T_est, model_points):
    """
    Calcola l'ADD (Average Distance of Model Points) tra due pose.

    Parametri:
    - R_gt: Matrice di rotazione ground truth (3x3)
    - T_gt: Vettore di traslazione ground truth (3,)
    - R_est: Matrice di rotazione stimata (3x3)
    - T_est: Vettore di traslazione stimata (3,)
    - model_points: array (N, 3) dei punti del modello
    
    Ritorna:
    - errore ADD (float)
    """
    #model_points_mm = model_points * 1000  # Non modifica l'originale

    #transformed_gt = (R_gt @ model_points.T).T + T_gt
    #transformed_est = (R_est @ model_points.T).T + T_est
    #distances = np.linalg.norm(transformed_gt - transformed_est, axis=1)
    T_gt = T_gt.reshape(3,) #* 1000         # Da metri a millimetri
    T_est = T_est.reshape(3,)# * 1000       # Da metri a millimetri
    transformed_gt = model_points @ R_gt.T + T_gt
    transformed_est = model_points @ R_est.T + T_est
    #print("traformed gt",transformed_gt)
    #print("traformed est",transformed_est)
    
    distances = np.linalg.norm(transformed_gt - transformed_est, axis=1)

    R_diff = R_gt.T @ R_est
    trace = np.trace(R_diff)
    cos_theta = (trace - 1) / 2
    # Clipping per evitare errori numerici fuori da [-1,1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = acos(cos_theta)
    theta_deg = degrees(theta_rad)
    add= np.mean(distances)*0.1
    diameter = compute_model_diameter(model_points)
    passed = add < (0.1 * diameter)
    return add,theta_deg,passed

def compute_adds_S(R_gt, T_gt, R_est, T_est, model_points):
    """
    Calcola l'ADD-S (simmetrico) per oggetti con simmetrie.

    Parametri:
    - R_gt: Matrice di rotazione ground truth (3x3)
    - T_gt: Vettore di traslazione ground truth (3,)
    - R_est: Matrice di rotazione stimata (3x3)
    - T_est: Vettore di traslazione stimata (3,)
    - model_points: array (N, 3) dei punti del modello

    Ritorna:
    - errore ADD-S (float)
    - errore angolare in gradi (float)
    - d: diametro modello (float)
    - passed: True se ADD-S < 0.1 * d
    """

    T_gt = T_gt.reshape(3,)
    T_est = T_est.reshape(3,)

    transformed_gt = model_points @ R_gt.T + T_gt
    transformed_est = model_points @ R_est.T + T_est

    # Nearest neighbor distance per punto stimato rispetto a GT
    kdtree = cKDTree(transformed_gt)
    distances, _ = kdtree.query(transformed_est, k=1)
    adds = np.mean(distances)

    # Errore angolare
    R_diff = R_gt.T @ R_est
    trace = np.trace(R_diff)
    cos_theta = (trace - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = acos(cos_theta)
    theta_deg = degrees(theta_rad)

    # Diametro modello
    d = np.max(cdist(model_points, model_points))
    passed = adds < 0.1 * d

    return adds, theta_deg, d, passed'''