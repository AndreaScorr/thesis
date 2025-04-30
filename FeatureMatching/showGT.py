

from PIL import Image,ImageDraw
import ImageUtils as img_utils
import json_utils as js
import cv2
import numpy as np
image_path1 = "/home/andrea/Desktop/Thesis_project/Inputs/000019.jpg"

image_id =  image_path1.split("/")[-1].removesuffix(".jpg").replace("0","")

image1_pil_show = Image.open(image_path1)

json_gt="/home/andrea/Downloads/ycbv_train_pbr/train_pbr/000049/scene_gt.json"
gt_R,gt_T=js.estrai_parametri(imgId=image_id,json_path=json_gt,target_obj_id=2)
print("GtR:",gt_R)
print("GtT:",gt_T)

fx = 1066.778
fy = 1067.487
cx = 312.9869079589844
cy = 241.3108977675438

#clfx,fy,cx,cy=intrinsic
cam_K = np.array([[fx,        0,      cx],
                    [0.0,       fy,     cy],
                          [0.0,       0.0,    1.0]])
        
tvec_gt = gt_T
rvec_gt, _ = cv2.Rodrigues(gt_R)

img_utils.draw_projected_3d_bbox(
image=image1_pil_show,
obj_id="2",
rvec=rvec_gt,
tvec=tvec_gt,
camera_matrix=cam_K,
dist_coeffs=None,
models_info_path="/home/andrea/Desktop/Thesis_project/Models/models_info.json"
)
