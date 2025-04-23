#import blenderproc as bproc

import numpy as np
import os
import sys
import cv2 as cv
from PIL import Image
import segmentation.segmentation_function as sf

sys.path.append(os.path.join(os.path.dirname(__file__), 'TRELLIS'))

from TRELLIS import modelGenerator as mg


image_path = "segmentation/data/omni6d/Inputs/scissors/000222.jpg"
mask_path  = "segmentation/Test/mask/"
mask_file_name = image_path.split("/")[-1]
segmentation_path = "segmentation/Test/segmentation"
segmentation_file_name = image_path.split("/")[-1]
text_prompt = "scissors"

    

sf.segmentation( image_path,mask_path,mask_file_name,segmentation_path,segmentation_file_name,text_prompt)


model_path = "model_test/mask"
mg.ModelGeneration(segmentation_path,model_path)



