from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os
import cv2 as cv

# load model
model = LangSAM()

assets_path = "data/omni6d/"
#inputs_path = assets_path + "Inputs/pencil"
#output_path = "output/segmentation/pencil"

inputs_path = assets_path + "Inputs/banana"
output_path = "output/segmentation/banana"
output_mask_path ="output/mask/pencil"
assets_amount = 0
#assets_data_type = ".png"
assets_data_type = ".jpg"

print(inputs_path)
for root_dir, cur_dir, files in os.walk(inputs_path):
    assets_amount += len(files)
print('file count:', assets_amount)

# Rename the files in the assets directory if they are not already named in the correct format
#rename_files_in_directory(inputs_path, assets_data_type)

#print_gpu_memory_every_sec()
# Load the LangSAM model and set the text prompt

model = LangSAM()
text_prompt = "ball"
#print_gpu_memory_every_sec()
for i in range(assets_amount):
    image_path = os.path.join(inputs_path, f"{str(i+1).zfill(5)}{assets_data_type}")
    image_pil = Image.open(image_path).convert("RGB")
    
    ### insert here the prompt ###

    #text_prompt = "ball"
    #text_prompt = "bottle"
    #text_prompt = "pencil"
    text_prompt="banana"
    print(f"Processing: {image_path}")

    results = model.predict([image_pil], [text_prompt])

    # Extract tje ,asl
    mask = results[0]["masks"]

    # Salva la maschera come immagine PNG
 

    # Convert to numpy array
    image_np = np.array(image_pil)  # (H, W, 3)
    mask_np = np.array(mask).astype(np.uint8).squeeze()  # (H, W)

    mask_output_path = os.path.join(output_mask_path, f"{str(i+1).zfill(5)}_mask.png")
    cv.imwrite(mask_output_path, mask_np * 255)

    # apply mask
    segmented_np = image_np * mask_np[:, :, None]  # Moltiplica per la maschera

    # convert to img
    segmented_pil = Image.fromarray(segmented_np, "RGB")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_mask_path,exist_ok=True)
    output_file = os.path.join(output_mask_path, f"{str(i+1).zfill(5)}{assets_data_type}")

    # save the image
    output_file = os.path.join(output_path, f"{str(i+1).zfill(5)}{assets_data_type}")
    segmented_pil.save(output_file)

    print(f"Saved: {output_file}")

    # Mostra l'immagine
