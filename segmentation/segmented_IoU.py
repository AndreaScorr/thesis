import argparse
import csv
import json
from PIL import Image
from lang_sam import LangSAM
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2 
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
    plt.show()
    '''plt.show(block=False)
    plt.pause(0.1)
    plt.close()'''
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
    mask_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if mask_np is None:
        raise FileNotFoundError(f"Immagine non trovata nel percorso: {path}")
    
    # Mostra l'immagine
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"Ground truth Mask")#: {id_str}_{obj_str}.png")
    plt.axis('off')
    plt.show()
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

        
def visualize_segmentations(mask_lang, mask_gt, image_path):
    """
    Visualizza l'immagine originale con le segmentazioni predetta e ground truth.

    Parameters:
    - mask_lang: np.ndarray — Maschera predetta (es. da LangSAM)
    - mask_gt: np.ndarray — Maschera ground truth
    - image_path: str — Percorso immagine RGB originale
    """
    # Carica immagine RGB
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Assicurati che le maschere siano uint8 (0 o 255)
    mask_gt = ((mask_gt > 0).astype(np.uint8)) * 255
    mask_lang = ((mask_lang > 0).astype(np.uint8)) * 255

    # Applica maschere all'immagine
    seg_gt = cv2.bitwise_and(img, img, mask=mask_gt)
    seg_lang = cv2.bitwise_and(img, img, mask=mask_lang)

    # Mostra i risultati
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(seg_gt)
    plt.title("GT Segmentation")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(seg_lang)
    plt.title("LangSAM Segmentation")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_segmentations_as_pdf(mask_lang, mask_gt, image_path, output_dir, image_id="sample"):
    """
    Salva immagine originale, segmentazione GT e segmentazione LangSAM come file PDF separati.

    Parameters:
    - mask_lang: np.ndarray — Maschera predetta (es. da LangSAM)
    - mask_gt: np.ndarray — Maschera ground truth
    - image_path: str — Percorso immagine RGB originale
    - output_dir: str — Cartella di destinazione per i PDF
    - image_id: str — Identificativo univoco per i nomi dei file
    """

    os.makedirs(output_dir, exist_ok=True)

    # Carica immagine
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepara maschere
    mask_gt = ((mask_gt > 0).astype(np.uint8)) * 255
    mask_lang = ((mask_lang > 0).astype(np.uint8)) * 255

    # Applica maschere
    seg_gt = cv2.bitwise_and(img, img, mask=mask_gt)
    seg_lang = cv2.bitwise_and(img, img, mask=mask_lang)

    # Funzione di salvataggio
    def save_image_pdf(image, title, filename):
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close()

    # Salvataggio immagini
    save_image_pdf(img, "Original Image", f"{image_id}_original.pdf")
    save_image_pdf(seg_gt, "GT Segmentation", f"{image_id}_gt.pdf")
    save_image_pdf(seg_lang, "Predicted Segmentation", f"{image_id}_pred.pdf")
    
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
#for id_image in ids[41:]: #[41:]:
id_image = 620

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
folder_str = str(int(id_folder)).zfill(6) 
id_str = str(int(id_image)).zfill(6)
obj_str = str(int(obj_id)).zfill(6)

path= f"/home/andrea/Desktop/test_set/ycbv_test_bop19/ycbv/test/{folder_str}/rgb/{id_str}.png"
visualize_segmentations(mask_lang, mask_gt, image_path=path)
output_dir="/home/andrea/Documents/thesis_material/Tomato"
save_segmentations_as_pdf(mask_lang, mask_gt, image_path=path,output_dir=output_dir, image_id=id_str)

IoU=compute_iou(prediction=mask_lang,
            ground_truth=mask_gt)

print("IoU:",IoU)
#os.makedirs(f"/home/andrea/Desktop/Thesis_project/segmentation/evaluation/csv/", exist_ok=True)
#csv_path = f"/home/andrea/Desktop/Thesis_project/segmentation/evaluation/csv/IoU.csv"
'''save_offset_result(csv_path=csv_path,
                scene_id=id_folder,
                im_id=id_image,
                obj_id=obj_id,
                IoU=IoU)
torch.cuda.empty_cache()
'''
