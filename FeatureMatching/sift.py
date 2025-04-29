import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt






def featureMatching(img_path1,img_path2):

    img1 = cv2.imread(img_path1, 0)  # queryImage
    #img1 = cv2.imread('/home/andrea/Desktop/Thesis_project/Inputs/000547.jpg', 0)  # queryImage
    img2 = cv2.imread(img_path2, 0)  # trainImage (segmentation mask)
    img_result=cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)

    # get images sizes
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # get the minimal size
    new_h = min(h1, h2)
    new_w = min(w1, w2)

    # Ridimensiona entrambe le immagini alla dimensione minima
    img1 = cv2.resize(img1, (new_w, new_h))
    img2 = cv2.resize(img2, (new_w, new_h))

    temp1,temp2=img1,img2
    img1 = cv2.Canny(img1,200,100)
    img2 = cv2.Canny(img2,200,100)


    # Get key point with orb
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create object BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # crossCheck=False perché lo faremo manualmente
    # Find matches
    matches = bf.knnMatch(des1, des2, k=2)  # Find the best 2 match for every descriptor

    #  each keypoint of the first image is matched with a number of keypoints from the second image.
    #  We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement).
    #  Lowe's test checks that the two distances are sufficiently different.
    #  If they are not, then the keypoint is eliminated and will not be used for further calculations.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Solo i match "buoni"
            good_matches.append(m)

    # order corrispondences by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Draw matches
    img3 = cv2.drawMatches(temp1, kp1, temp2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img4 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])  # [N, 2]
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])  # [N, 2]

    if len(good_matches) > 4:
        # Estrai i punti corrispondenti
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calcola l’omografia con RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # mask è un array [1,1,1,0,1,...] che indica i match inlier (quelli buoni)
        matchesMask = mask.ravel().tolist()

        # Disegna solo gli inlier
        img_inliers = cv2.drawMatches(
            temp1, kp1, temp2, kp2, good_matches,
            None,
            matchColor=(0, 255, 0),  # Verde = inlier
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        plt.title("Inlier Matches (RANSAC)")
        plt.imshow(img_inliers)
        plt.show()

    else:
        print("Not enough matches for RANSAC")

    plt.imshow(img3)
    plt.show()

    return points1 , points2



'''
img_path1= "/home/andrea/Desktop/Thesis_project/Segmented/rgb/banana_000019.jpg"
img_path2 = "/home/andrea/Desktop/Thesis_project/blender_render/obj_000010.ply/000001.png"

points1,points2= featureMatching(img_path1,img_path2)

print(points1[0].shape)

img1_pixel_matches=[]

#  Extract the pixels corresponding to the good point
for match in good_matches:
    # Estrai le coordinate dei punti chiave corrispondenti
    img1_pt = kp1[match.queryIdx].pt  # Punto corrispondente nell'immagine 1
    img2_pt = kp2[match.trainIdx].pt  # Punto corrispondente nell'immagine 2
    img1_pixel_matches.append(img1_pt)
'''