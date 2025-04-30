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
import json_utils as js


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





def find_correspondences(image_path1: str, image_path2: str, num_pairs: int = 10, load_size: int = 224, layer: int = 8,
                         facet: str = 'token', bin: bool = True, thresh: float = 0.20, model_type: str = 'dino_vits8',
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
    #device = 'cpu'
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
            # Debug: stampa indici dei patch (float)
        #print(f"Raw patch indices -> img1: (y={y1:.2f}, x={x1:.2f}), img2: (y={y2:.2f}, x={x2:.2f})")

        # Debug: stampa parametri del modello
        #print(f"Stride: {extractor.stride}, Patch size: {extractor.p}")

        x1_show = (int(x1) -1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y1_show = (int(y1) -1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        x2_show = (int(x2) -1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y2_show = (int(y2) -1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2

        #print(f"Final coords -> img1: (y={y1_show}, x={x1_show}), img2: (y={y2_show}, x={x2_show})\n")

        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))
    return points1, points2, image1_pil, image2_pil


def draw_correspondences(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                         image1: Image.Image, image2: Image.Image) -> Tuple[plt.Figure, plt.Figure]:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: two figures of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)
    fig1, ax1 = plt.subplots()
    ax1.axis('off')
    fig2, ax2 = plt.subplots()
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                               "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 8, 1
    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)
    return fig1, fig2

def segmentation(image_pil ,text_prompt):
    """
    :param str image_path

    """
    model = LangSAM()
    results = model.predict([image_pil], [text_prompt])

    #extract the mask
    mask = results[0]["masks"]


    # Convert to numpy array
    image_np = np.array(image_pil)  # (H, W, 3)
    mask_np = np.array(mask).astype(np.uint8).squeeze()  # (H, W)
    mask_np = (mask_np > 0).astype(np.uint8) * 255  # adesso è 0 o 255
    ''''''
    # Fix maschera se necessario
    if mask_np.ndim == 3:
        if mask_np.shape[2] == 3:
            mask_np = mask_np[:, :, 0]  # Prendi solo il primo canale
        elif mask_np.shape[0] == 1:
            mask_np = mask_np.squeeze(0)

    width, height = image_pil.size
    #plt.imshow(mask_np,cmap="gray")

    '''
    apply mask
    '''

    #torch.cuda.empty_cache()
    return mask_np

def get_bounding_box_from_mask(mask):
    # Convert to binary mask (0 and 1) if it is not
    mask_binary = np.where(mask > 0, 1, 0)

    # Find min and max rows and columns with a value of 1
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # return top-left and bottom-right corners and width, height
    x_left = cmin
    y_upper = rmin
    w = cmax - cmin + 1
    h = rmax - rmin + 1
    
    return [x_left, y_upper, w, h]


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











if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    config = load_config(args.config)

    with torch.no_grad():

        # prepare directories
        #root_dir = Path(args.root_dir)
        #pair_dirs = [x for x in root_dir.iterdir() if x.is_dir()]
        save_dir = "temp"
        #save_dir.mkdir(exist_ok=True, parents=True)

        
        image_path1 = "/home/andrea/Desktop/Thesis_project/Inputs/000547.jpg"
        #image_path1 = "/home/andrea/Desktop/Thesis_project/Inputs/000019.jpg"
        
        image_id =  image_path1.split("/")[-1].removesuffix(".jpg").replace("0","")
        #print("image_id:",image_id)
        #image_path1 = "/home/andrea/Desktop/Thesis_project/Inputs/000305.jpg"
        
        image_path2 = "blender_render/obj_000014.ply/000001.png"
        

        image1_pil =Image.open(image_path1).convert('RGB')
        image1_pil_show = Image.open(image_path1)
        plt.figure(2)
        plt.imshow(image1_pil)
        plt.show()
        mask = segmentation(image1_pil,config["text_prompt"])

        

        bbox = get_bounding_box_from_mask(mask)

        img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(image1_pil), bbox)

        print("imgcrop",img_crop.shape)
        print("y_offset:" ,y_offset)
        print("x_offset:",x_offset)
        mask_crop,_,_ = img_utils.make_quadratic_crop(mask, bbox)
        mask_crop =mask_crop[:,:,0]
        mask_crop = (mask_crop > 0).astype(np.uint8) * 255  # adesso è 0 o 255

        #print("mask crop", mask_crop.shape)



        img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
        
        #print("imgcrop size:",img_crop.shape)
        plt.figure(3)
        plt.imshow(img_crop)
        
        plt.show()
        # compute point correspondences
        
        image_path1="temp/img_crop.png"
        cv2.imwrite(image_path1,img_crop)

        points1, points2, image1_pil, image2_pil = find_correspondences(image_path1,image_path2,num_pairs=20)

        print("points1:",points1)
        
        print("points2:",points2)
        draw_correspondences(points1,points2,image1_pil,image2_pil)
        #plt.show()
        
        
        _,y_offset,x_offset=img_utils.make_quadratic_crop(image1_pil,bbox)
        

        print("image1:",image1_pil.size)
        print("image2:",image2_pil.size)
        
        

        i=1
        nocs_path = "/home/andrea/Desktop/Thesis_project/nocs/obj_000014.ply" #config["mask_path"].replace("/Segmented/mask","/nocs")+"/"+text_prompt+"_"+subfolder_nocs+"_sample"
        assets_data_type =".png"
        image_nocs_path = os.path.join(nocs_path, f"{str(i).zfill(6)}{assets_data_type}")
        #print(f'image_nocs path: {image_nocs_path}')
        img_nocs =cv2.imread(image_nocs_path,cv2.IMREAD_COLOR_RGB)
        #img_nocs= cv2.resize(img_nocs,(224,224))
        #img_nocs = cv2.resize(img_nocs, (image2_pil.shape[0], image2_pil.shape[1]))
        img_nocs = cv2.resize(img_nocs, image2_pil.size)
        
        #print("image_nocs" ,img_nocs.shape)
        

        assets_data_type= ".png"
        #print(img2_pixel_matches)
        features3Dpoint =[]
        for x,y in points2:
            x1, y1 = int(y), int(x)  # Convert to int
            ##print(x)
            #cv2.circle(img_nocs,(x1,y1), 1, (0, 255, 0), 2)  # (x, y), raggio, colore (BGR), spessore
            if np.all(img_nocs[y1, x1] == [1, 1, 1]) or np.all(img_nocs[y1, x1] == [0, 0, 0]):
                x1,y1,new_color = img_utils.find_nearest_non_black_white(img_nocs,x1,y1,3)
            else:
                new_color = img_nocs[y1, x1]
            #features3Dpoint.append(img_nocs[y1,x1])
            features3Dpoint.append(tuple(map(int, new_color))) #use the color of the match pixel

            ##print(f'posizione {x1,y1} colore: {img_nocs[y1,x1]}')
            img_nocs[y1,x1]=(255,0,0) #paint the pixel of the match

        plt.figure(5)
        plt.imshow(img_nocs)

        plt.show()

        
         #Converti in array NumPy
        points1 = np.array(points1)

        print("y_offset " , y_offset)
        print("x_offset: ", x_offset)

        points1[:, 0] += y_offset  # Aggiungi l'offset alla prima colonna (x)
        points1[:, 1] += x_offset  # Aggiungi l'offset alla prima colonna (x)
        points1[:,[0,1]] = points1[:,[1,0]]


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

        print(image1_pil_show.size)
        print(image1_pil.size)
        
        target_size = (image1_pil.size[1],image1_pil.size[0]) 
        fx = 1066.778
        fy = 1067.487
        cx = 312.9869079589844
        cy = 241.3108977675438

        _,intrinsic=img_utils.input_resize(image1_pil_show,image1_pil.size,intrinsics=[fx, fy, cx, cy])
        print("instrinsic:",intrinsic)
        #clfx,fy,cx,cy=intrinsic
        #fx,fy,cx,cy=intrinsic
        cam_K = np.array([[fx,        0,      cx],
                          [0.0,       fy,     cy],
                          [0.0,       0.0,    1.0]])
        
        print("new CAm",cam_K)
        object_points_3D = np.array(features_nocs_to_mesh, dtype=np.float32)
        image_points_2D = np.array(points1, dtype=np.float32)
        


        #success,rvec,tvec=cv2.solvePnP(object_points_3D, image_points_2D, cam_K, distCoeffs=None)
        dist_coeffs = np.ones((5, 1)) *0

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points_3D, image_points_2D, cam_K,distCoeffs=dist_coeffs, iterationsCount=200, reprojectionError=9.0)
        
        
        R, _ = cv2.Rodrigues(rvec)
        
        tvec_gt = np.array([94.04884338, -50.48897934, 572.37988281])
        #tvec=tvec_gt
        error= np.linalg.norm(tvec-tvec_gt)
        distance = np.sqrt(np.sum((tvec - tvec_gt)**2))
        print("error:",error)
        print("distance ^2: ", distance)
        print("R:",R)
        print("t:",tvec)
        
        json_gt="/home/andrea/Downloads/ycbv_train_pbr/train_pbr/000049/scene_gt.json"
        gt_R,gt_T=js.estrai_parametri(imgId=image_id,json_path=json_gt,target_obj_id=14)
        print("GtR:",gt_R)
        print("GtT:",gt_T)

        rvec_gt,_=cv2.Rodrigues(gt_R)
        tvec_gt = gt_T
        img_utils.draw_projected_3d_bbox(
        image=image1_pil_show,
        obj_id="14",
        rvec=rvec,
        tvec=tvec,
        camera_matrix=cam_K,
        dist_coeffs=None,
        models_info_path="/home/andrea/Desktop/Thesis_project/Models/models_info.json"
        )
        