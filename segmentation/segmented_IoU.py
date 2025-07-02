import argparse
import csv
import json
from PIL import Image
from lang_sam import LangSAM
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2 as cv
import yaml
import torch


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def Lang_segmentation(id_folder,id_image,text_prompt):
    """
    :param str image_path

    """
    folder_str = str(int(id_folder)).zfill(6) 
    id_str = str(int(id_image)).zfill(6)
    
     # Leggi il file JSON
    with open('/home/andrea/Desktop/Thesis_project/segmentation/remapping_segmentation.json', 'r') as f:
        folders = json.load(f)

   
    model = LangSAM()
    
    image_path=f"/home/andrea/Desktop/test_set/ycbv_test_bop19/ycbv/test/{folder_str}/rgb/{id_str}.png"
    image_pil = Image.open(image_path).convert("RGB")
    try:
        results = model.predict([image_pil], [text_prompt])
    except:
        return np.zeros((image_pil.size[1],image_pil.size[0]))
    #extract the mask
    mask = results[0]["masks"]#[1]

    
    mask_np = np.array(mask).astype(np.uint8)#.squeeze()  # (H, W)
    # Se è 3D, scelgo la prima maschera
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    elif mask_np.ndim == 2:
        mask_np = mask_np
    else:
        #raise ValueError(f"Maschera con shape inaspettata: {mask_np.shape}")
        return np.zeros((image_pil.size[1],image_pil.size[0]))


    # Visualizza la maschera
    plt.imshow(mask_np, cmap='gray')  # Usa 'gray' per mostrare una maschera binaria
    plt.title("Predicted Mask")
    plt.axis('off')  # Rimuove gli assi
    #plt.show()
    plt.show(block=False)
    plt.pause(0.1)
    plt.close() 
    #cv.imwrite(mask_path, mask_np * 255)




    #os.makedirs(segmentation_path, exist_ok=True)
    
    # save the image
    #output_file = os.path.join(segmentation_path, segmentation_file_name )
    #segmented_pil.save(output_file)

    #print(f"Saved: {output_file}")
    #torch.cuda.empty_cache()
    return mask_np

def gt_segmentation(id_folder,id_image,obj_id):
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
    mask_np = cv.imread(path, cv.IMREAD_GRAYSCALE)
    
    if mask_np is None:
        raise FileNotFoundError(f"Immagine non trovata nel percorso: {path}")
    
    # Mostra l'immagine
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"Mask: {id_str}_{obj_str}.png")
    plt.axis('off')
    #plt.show()'''
    '''plt.show(block=False)
    plt.pause(1)
    plt.close()'''
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

def compute_iou(prediction, ground_truth):
    intersection = np.logical_and(prediction, ground_truth).sum()
    union = np.logical_or(prediction, ground_truth).sum()
    iou = intersection / union if union != 0 else 1.0  # 1.0 se entrambi vuoti
    return iou

def save_offset_result(csv_path, scene_id, im_id, obj_id,IoU):

    
    # Crea la riga
    row = {
        'scene_id': scene_id,
        'im_id': im_id,
        'obj_id': obj_id,
        'IoU': IoU
        
    }

    # Scrittura (header solo se il file non esiste)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene_id', 'im_id', 'obj_id', 'IoU' ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

        

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()

config = load_config(args.config)

#image_path="/home/andrea/Desktop/test_set/ycbv_test_bop19/ycbv/test/000048/rgb/000001.png"#config["image_path"]

id_folder = 50
json_path= f"/home/andrea/Desktop/test_set/ycbv_test_bop19/ycbv/test/{str(id_folder).zfill(6)}/scene_gt.json"
print(json_path)
with open(json_path, 'r') as f:
    data = json.load(f)
ids = list(map(int, data.keys()))
text_prompt = config["text_prompt"]
for id_image in ids[41:]: #[41:]:
#id_image = 1

    obj_id = config["obj_id"]
    #mask_file_name = text_prompt+ "_"+ image_path.split("/")[-1]
    #segmentation_file_name = text_prompt+ "_"+image_path.split("/")[-1]
    obj_str = str(int(obj_id)).zfill(6) 
    segmentation_file_name = text_prompt+ "_"+obj_str

    mask_lang = Lang_segmentation(id_folder=id_folder,
                                id_image=id_image,
                                text_prompt=text_prompt)
    mask_gt=gt_segmentation(id_folder=id_folder,
                    id_image=id_image,
                    obj_id=obj_id)

    IoU=compute_iou(prediction=mask_lang,
                ground_truth=mask_gt)
    
    print("IoU:",IoU)
    os.makedirs(f"/home/andrea/Desktop/Thesis_project/segmentation/evaluation/csv/", exist_ok=True)
    csv_path = f"/home/andrea/Desktop/Thesis_project/segmentation/evaluation/csv/IoU.csv"
    save_offset_result(csv_path=csv_path,
                    scene_id=id_folder,
                    im_id=id_image,
                    obj_id=obj_id,
                    IoU=IoU)
    torch.cuda.empty_cache()

