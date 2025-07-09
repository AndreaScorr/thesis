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
import Evaluation_utils as Eval
import findBest_template as fbt

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
    #print("size num_patches1:",(num_patches1))
    #print("size num_patches2:",(num_patches2))

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
    #ax1.axis('off')
    fig2, ax2 = plt.subplots()
    #ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)

    # Sposta le finestre (valori in pixel: x, y)
    fig1.canvas.manager.window.wm_geometry("+100+100")   # Prima figura in alto a sinistra
    fig2.canvas.manager.window.wm_geometry("+800+100")   # Seconda figura più a destra

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
    mask = results[0]["masks"][0]


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


def segmentation_from_file(id_folder,id_image,obj_id):
    folder_str = str(int(id_folder)).zfill(6) 
    id_str = str(int(id_image)).zfill(6)
    
     # Leggi il file JSON
    with open('/home/andrea/Desktop/Thesis_project/segmentation/remapping_segmentation.json', 'r') as f:
        folders = json.load(f)

    # Funzione per ottenere gli oggetti di una cartella
    def get_objects_by_folder_id(folder_id):
        for folder in folders:
            if folder["id_folder"] == folder_id:
                return folder["objects"]
        return None

    # Esempio di utilizzo
    #id_da_cercare = 54
    oggetti = get_objects_by_folder_id(id_folder)
    remap =oggetti.index(obj_id)
    #print("oggetti",type(oggetti))
    #print("remap",remap)
    obj_id = remap

    obj_str = str(int(obj_id)).zfill(6)
    #path=f"/home/andrea/test_set/ycbv_test_all/test/{folder_str}/mask_visib/{id_str}_{obj_str}.png"
    path= f"/home/andrea/test_set/ycbv_test_all/ycbv/test/{folder_str}/mask_visib/{id_str}_{obj_str}.png"
    # Leggi l'immagine
    mask_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # Definisci il kernel: controlla quanto "smussa"
    kernel = np.ones((3, 3), np.uint8)
    mask_np = cv2.erode(mask_np, kernel, iterations=1)

    if mask_np is None:
        raise FileNotFoundError(f"Immagine non trovata nel percorso: {path}")
    
    # Mostra l'immagine
    #plt.imshow(mask_np, cmap='gray')
    #plt.title(f"Mask: {id_str}_{obj_str}.png")
    #plt.axis('off')
    #plt.show()
    
    #plt.show(block=False)
    #plt.pause(3)
    #plt.close() 
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





def Estimate_Pose_from_correspondences(id_folder,id_image, file_type, template_id,best_templates, obj_id):
        """
        Calculate the pose 
        Parameters:
        - id_image: id of the input image
        - file_type: type of the input image png or jpg
        - template_id: it the template id used for feature matching
        - obj_id: is the object id from 1 to 21 
        """
        print("id_folder",id_folder)
        print("id_imag",id_image)
        print("obj_id",obj_id)
        

        save_dir = "temp"
        #save_dir.mkdir(exist_ok=True, parents=True)

        
        #id_image = 201
        folder_str = str(int(id_folder)).zfill(6) 
        id_str = str(int(id_image)).zfill(6)
        min_rotation_error=np.inf
        min_translation_error = np.inf
        best_template_id =np.inf

        
  
        #file_type = "jpg"

        #image_path1 = f"/home/andrea/test_set/ycbv_test_all/test/{folder_str}/rgb/{id_str}.{file_type}"  
        image_path1 = f"/home/andrea/test_set/ycbv_test_all/ycbv/test/{folder_str}/rgb/{id_str}.{file_type}"
        #image_path1 = "/home/andrea/Desktop/Thesis_project/Inputs/000106.jpg"
        
        image_id =  image_path1.split("/")[-1].removesuffix(f".{file_type}")
        #image_id =  image_path1.split("/")[-1].removesuffix(".png")

        image_id = int(image_id)
        image1_pil =Image.open(image_path1).convert('RGB')
        image1_pil_show = Image.open(image_path1)
        #plt.imshow(image1_pil_show)
        #plt.show()
        
        best_t_id = 0
        
        obj_str = str(int(obj_id)).zfill(6)  # Garantisce che obj_id sia sempre a 6 cifre con zeri iniziali
        
        #image_path2 = f"blender_render/obj_{obj_str}.ply/{str(template_id).zfill(6)}.png"  
        #image_path2 = f"blender_render/obj_{obj_str}.ply/{str(t).zfill(6)}.png"  
        

        
        

        


        image1_pil =Image.open(image_path1).convert('RGB')
        image1_pil_show = Image.open(image_path1)
        #plt.figure(2)
        #plt.imshow(image1_pil)
        #plt.show()
        #plt.show(block=False)
        #plt.pause(1)
        #plt.close()
        #mask = segmentation(image1_pil,config["text_prompt"])
        mask = segmentation_from_file(id_folder=id_folder,id_image=id_image,obj_id=obj_id)

        '''
        try:
            mask = segmentation(image1_pil,config["text_prompt"])
        except:
            mask = segmentation_from_file(id_folder=id_folder,id_image=id_image,obj_id=obj_id) '''
        
        bbox = get_bounding_box_from_mask(mask)

        #print("image1_pil prima crop:",image1_pil.size)
    
        
        img_crop, y_offset, x_offset = img_utils.make_quadratic_crop(np.array(image1_pil_show), bbox)

        #print("imgcrop",img_crop.shape)
        
    

        #print("y_offset:" ,y_offset)
        #print("x_offset:",x_offset)
        mask_crop,_,_ = img_utils.make_quadratic_crop(mask, bbox)
        mask_crop =mask_crop[:,:,0]
        mask_crop = (mask_crop > 0).astype(np.uint8) * 255  # adesso è 0 o 255

        #print("mask crop", mask_crop.shape)
        print("img_crop shape:", img_crop.shape)
        print("mask_crop shape:", mask_crop.shape)


        img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
        buffer= "/home/andrea/Desktop/Thesis_project/FeatureMatching/buffer/crop.png"
        cv2.imwrite(buffer,img_crop)
        best_template=fbt.find_Best_template_patchwise(input_image_path=buffer,template_dir=f"/home/andrea/Desktop/ZS6/ZS6D/templates/ycbv_desc/obj_{str(obj_id)}")
        best_template_id=int(best_template.removesuffix(".png"))
        #qui andrà la modifica, si cercherà nel template di philip
        #caricare l'immagine corrispondente uv
        #caricare i normfactor

        '''cv2.imshow("Cropped Image", img_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                    img_crop= cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        #print("imgcrop size:",img_crop.shape)
        plt.figure(3)
        plt.axis("off")
        plt.imshow(img_crop)
        plt.show()'''
        #plt.show(block=False)
        #plt.pause(1)
        #plt.close()
        # compute point correspondences
        
        image_path1="temp/img_crop.png"
        cv2.imwrite(image_path1,img_crop)
        image_path2=f"/home/andrea/Desktop/ZS6/ZS6D/templates/ycbv_desc/obj_{str(obj_id)}/{best_template}"
        print("image_path2:",image_path2)
        img_uv_path =image_path2.removesuffix(".png")
        img_uv_path = img_uv_path+"_uv.npy"
        print(img_uv_path)
       
        img_uv = np.load(img_uv_path)
        img_uv = img_uv.astype(np.uint8)
        img_uv = cv2.resize(img_uv, (img_crop.shape[1], img_crop.shape[0]))
        print("img_uv shape:", img_uv.shape)
        print("img_crop shape:", img_crop.shape)
        #img_uv = cv2.imshow("image uv",img_uv)
                # Aspetta che una chiave venga premuta
        #cv2.waitKey(0)

        # Chiude la finestra dopo la pressione di un tasto
        #cv2.destroyAllWindows()
        print("best_template_path",image_path2)
        try:
            points1, points2, image1_pil, image2_pil = find_correspondences(image_path1,image_path2,load_size=img_crop.shape[0],num_pairs=30) #30 #original
        except:
            result = {
            "id_image": id_image,
            "data": {
                "GT_R": None,
                "GT_T": None,
                "ADD": 0,
                "ADD_S": 0
                }
            }
            return result

        print("n_correspondences:",len(points1))
        if((len(points1))<=4):
            result = {
            "id_image": id_image,
            "data": {
                "GT_R": None,
                "GT_T": None,
                "ADD": 0,
                "ADD_S": 0
                }
            }
            return result
        #print("image1_pil dopo crop:",image1_pil.size)



        
        original_size_x,original_size_y= image1_pil.size
        diag_original = np.sqrt(original_size_x**2+original_size_y**2)


        crop_size_x,crop_size_y= img_crop[:,:,0].shape
        diag_crop = np.sqrt(crop_size_x**2+crop_size_y**2)
        size_ratio_x = original_size_x/crop_size_x
        size_ratio_y = original_size_y/crop_size_y
        scale_ratio = diag_original/diag_crop

        #print("size ratio x:",size_ratio_x)
        #print("size ratio y:",size_ratio_y)
        #print("points1:",points1)
        
        #print("points2:",points2)
        #draw_correspondences(points1,points2,image1_pil,image2_pil)
        #plt.show()
        '''plt.show(block=False)
        plt.pause(1)
        plt.close()
        
        plt.show(block=False)
        plt.pause(1)
        plt.close()'''
        
        _,y_offset,x_offset=img_utils.make_quadratic_crop(image1_pil_show,bbox)
        #print("y_offset after crop show:" ,y_offset)
        #print("x_offset after crop:",x_offset)

        #print("image1:",image1_pil.size)
        #print("image2:",image2_pil.size)
        fx = 1066.778
        fy = 1067.487
        cx = 312.9869 #312.9869079589844
        cy = 241.3109 #241.3108977675438

        cam_K = np.array([[fx,         0,      cx],
                        [0.0,        fy,     cy],
                        [0.0,       0.0,    1.0]])
        resize_factor_path = "/home/andrea/Desktop/ZS6/ZS6D/templates/ycbv_desc/models_xyz/norm_factor.json"

        with open(resize_factor_path, 'r') as f:
            norm_factors = json.load(f)
        try:
            R_est, t_est = img_utils.get_pose_from_correspondences(points1, points2, 
                                                                   y_offset, x_offset, 
                                                                   img_uv, cam_K, 
                                                                   norm_factors[str(obj_id)], 
                                                                   scale_factor=1.0, 
                                                                   resize_factor=1.0)
            print("R_est",R_est)
            print("t_est",t_est)
        except:
            result = {
            "id_image": id_image,
            "data": {
                "GT_R": None,
                "GT_T": None,
                "ADD": 0,
                "ADD_S": 0
                }
            }
            return result
        
        print("R_est",R_est)
        print("t_est",t_est)

        json_gt=f"/home/andrea/Desktop/test_set/ycbv_test_bop19/ycbv/test/{folder_str}/scene_gt.json"
        gt_R,gt_T=js.estrai_parametri(imgId=image_id,json_path=json_gt,target_obj_id=obj_id)
        gt_T =np.asarray(gt_T).reshape(-1)
        
        print("GtR:",gt_R)
        print("GtT:",gt_T)
        print("rotation error" , Eval.rotation_error(gt_R,R_est))
        error = Eval.translation_error(gt_T, t_est)
        print(f"Errore di traslazione: {error:.2f} mm")

        tvec=t_est
        rvec,_=cv2.Rodrigues(R_est)
        rvec = np.asarray(rvec, dtype=np.float64).reshape((3, 1))


        rvec_gt,_=cv2.Rodrigues(gt_R)
        tvec_gt = gt_T

        img_utils.draw_projected_3d_bbox_gt(
        folder_id=id_folder,
        image_id= image_id,
        image=image1_pil_show,
        obj_id=str(obj_id),
        rvec=rvec,
        tvec=tvec,
        rvec_gt=rvec_gt,
        tvec_gt=tvec_gt,
        camera_matrix=cam_K,
        dist_coeffs=None,
        models_info_path="/home/andrea/Desktop/Thesis_project/Models/models_info.json"
        )

        points_eval =np.array(img_utils.transform_2D_3D(points2,img_uv, norm_factors[str(obj_id)]))
        best_d= Eval.compute_model_diameter(model_points=points_eval,obj_id=obj_id,models_info_path="/home/andrea/Desktop/Thesis_project/Models/models_info.json")

        Add,AddS = Eval.compute_add_and_addS(folder=id_folder,
                                        id_image=image_id,
                                        obj_id=obj_id,
                                        pts3d=points_eval,
                                        diameter=best_d,
                                        R_gt=gt_R,
                                        t_gt=gt_T,
                                        R_pred=R_est,
                                        t_pred=t_est,
                                        best_template_id=best_template_id)
        print("ADD:",Add)
        #print("% 1  diameter:",(d*0.01))
        #print("theta:", theta)
       
        
        print("ADD-S:",AddS)

        result = {
        "id_image": id_image,
        "data": {
            "GT_R": gt_R,
            "GT_T": gt_T,
            "ADD": Add,
            "ADD_S": AddS
            }
        }
        #torch.cuda.empty_cache()
        return result
    
            






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    config = load_config(args.config)

    with torch.no_grad():
        obj_id = config["obj_id"]
        best=[]
        best_template_id = [89]
        template_id =17
        # Load JSON file
        #id_folder=50
        #folder_str = str(int(id_folder)).zfill(6)
        #json_path= f"/home/andrea/Desktop/test_set/ycbv_test_bop19/ycbv/test/{folder_str}/scene_gt.json"
        '''with open(json_path, 'r') as f:
            data = json.load(f)
        ids = list(map(int, data.keys()))
        print(ids)
        check_ids =[83, 1027, 1059, 1087, 1568, 1576, 2051] #49 obj 6 [1172, 2061]# 48 [1128,1122, 1137]'''
#        for id_image in range(1,2):
        with open('/home/andrea/Desktop/Thesis_project/segmentation/remapping_segmentation.json', 'r') as f:
            data = json.load(f)

        folders_to_fetch=[]
        #for id_folder in range(48,60):
        for entry in data:
            objects=entry["objects"]
            #print(objects)
            #print(obj_id in objects)
            if obj_id in objects:
                folders_to_fetch.append(entry["id_folder"])
                #if obj_id in 
        
        print(folders_to_fetch)
            
        for id_folder in folders_to_fetch:
            folder_str = str(int(id_folder)).zfill(6)
            json_path= f"/home/andrea/Desktop/test_set/ycbv_test_bop19/ycbv/test/{folder_str}/scene_gt.json"
            with open(json_path, 'r') as f:
                data = json.load(f)
            ids = list(map(int, data.keys()))
            print(ids)
            
            #check_ids =[83, 1027, 1059, 1087, 1568, 1576, 2051] #49 obj 6 [1172, 2061]# 48 [1128,1122, 1137]
            for id_image in ids: #check_ids:#

                #for template_id in best_template_id:
                #id_image=8
                    #try:
                #print("templateooo",template_id)

                dict_st_result=Estimate_Pose_from_correspondences(id_folder=id_folder,
                                                                id_image=id_image,
                                                                file_type="png",
                                                                template_id=template_id,
                                                                best_templates=best_template_id,
                                                                obj_id=obj_id)
                if(dict_st_result["data"]["ADD"]==1 or dict_st_result["data"]["ADD_S"]==1):
                    best.append(id_image)
                    print("template",template_id)
                print(dict_st_result)
                    #except:
                    #    continue
            print(best)
            