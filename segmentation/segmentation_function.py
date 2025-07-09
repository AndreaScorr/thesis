from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os
import cv2 as cv
import yaml
import argparse
#import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def segmentation(image_path , mask_path , mask_file_name, segmentation_path, segmentation_file_name ,text_prompt):
    """
    :param str image_path

    """
    model = LangSAM()
    image_pil = Image.open(image_path).convert("RGB")
    results = model.predict([image_pil], [text_prompt])

    #extract the mask
    mask = results[0]["masks"]#[1]

    folder_id = image_path.split("/")[-3].replace("0","")
    print(folder_id)
    # Convert to numpy array
    image_np = np.array(image_pil)  # (H, W, 3)
   

    mask_np = np.array(mask).astype(np.uint8).squeeze()  # (H, W)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    elif mask_np.ndim == 2:
        mask_np = mask_np

    os.makedirs(mask_path, exist_ok=True)
    mask_path = os.path.join(mask_path, mask_file_name)
    cv.imwrite(mask_path, mask_np * 255)



    # apply mask
    segmented_np = image_np * mask_np[:, :, None]  # Moltiplica per la maschera

    # convert to img
    segmented_pil = Image.fromarray(segmented_np, "RGB")

    os.makedirs(segmentation_path, exist_ok=True)
    
    # save the image
    output_file = os.path.join(segmentation_path, segmentation_file_name )
    segmented_pil.save(output_file)

    print(f"Saved: {output_file}")
    #torch.cuda.empty_cache()
    return True


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()

config = load_config(args.config)
id_image = 2070
id_image_str=(str(id_image).zfill(6))

scene_id=48
scene_id_str=(str(scene_id).zfill(6))
image_path= f"/home/andrea/Desktop/test_set/ycbv_test_bop19/ycbv/test/{scene_id_str}/rgb/{id_image_str}.png"  #config["image_path"]

mask_path= f"/home/andrea/Desktop/Thesis_project/Segmented/mask/{scene_id_str}/{id_image_str}.png" #config["mask_path"]

segmentation_path =f"/home/andrea/Desktop/Thesis_project/Segmented/rgb/{scene_id_str}/" #config["segmentation_path"]

text_prompt = config["text_prompt"]

obj_id = config["obj_id"]
obj_str = str(int(obj_id)).zfill(6) 

mask_file_name = id_image_str+"_"+obj_str+".png"
#segmentation_file_name = text_prompt+ "_"+image_path.split("/")[-1]
segmentation_file_name = id_image_str+"_"+obj_str+".png"
print(config)

'''
image_path = "segmentation/data/omni6d/Inputs/banana/00001.jpg"
mask_path  = "segmentation/Test/mask/"
mask_file_name = image_path.split("/")[-1]
segmentation_path = "segmentation/Test/segmentation"
segmentation_file_name = image_path.split("/")[-1]
text_prompt = "banana"
'''
    
    

segmentation( image_path,mask_path,mask_file_name,segmentation_path,segmentation_file_name,text_prompt)