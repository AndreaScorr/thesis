

import cv2
import numpy as np
from scipy import spatial
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
from sklearn.neighbors import KDTree

import csv
import os

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


def add(pts,R_gt, T_gt, R_est, T_est,obj_id,diameter,percentage=0.1):
    #with open(models_info_path, "r") as f:
    #    models_info = json.load(f)
    center_dist = np.linalg.norm(T_est - T_gt)
    #spheres_overlap = center_dist < models_info[str(obj_id)]["diameter"]
    #print("spheres_overlap:",spheres_overlap)

    pts_est = transform_pts_Rt(pts, R_est, T_est)
    pts_gt = transform_pts_Rt(pts, R_gt, T_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    #print("err:",e)
    #return e
    threshold = diameter * percentage
    print("diameter",diameter)
    print("mean distance:",e)
    print("threshol",threshold)
    # Score binario
    score = 1 if e < threshold else 0
    return score

def compute_add_score_single(pts3d, diameter, R_gt, t_gt, R_pred, t_pred , percentage=0.1):
    #print("pts3D ", pts3d[:5])
    #pts3d = pts3d / 1000
    #print("pts3D ", pts3d[:5])
    
    #t_gt = t_gt / 1000
    #t_pred = t_pred / 1000
    
    
    # Trasformazione dei punti 3D con le pose
    pts_xformed_gt = R_gt @ pts3d.T + t_gt.reshape(3, 1)
    pts_xformed_pred = R_pred @ pts3d.T + t_pred.reshape(3, 1)

    # Calcolo della distanza media
    distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
    mean_distance = np.mean(distance)
    #diameter = diameter/1000
    # Soglia
    threshold = diameter * percentage
    print("diameter",diameter)
    print("mean distance:",mean_distance)
    print("threshol",threshold)
    # Score binario
    score = 1 if mean_distance < threshold else 0
    return score

def translation_error(t_gt, t_pred):
    #print("DEkta",t_gt - t_pred)
    return np.linalg.norm(t_gt - t_pred)    

def compute_adds_score(pts3d, diameter, R_gt, t_gt, R_pred, t_pred, percentage=0.1):
    #pts3d = pts3d / 1000
    #t_gt = t_gt / 1000
    #t_pred = t_pred / 1000
    
    # Se la predizione è NaN, ritorna 0
    if np.isnan(np.sum(t_pred)):
        return 0

    print("diameter:",diameter)
    # Trasformazione dei punti
    pts_xformed_gt = R_gt @ pts3d.T + t_gt.reshape(3, 1)
    pts_xformed_pred = R_pred @ pts3d.T + t_pred.reshape(3, 1)

    # Distanza media punto più vicino (ADD-S)
    kdt = KDTree(pts_xformed_gt.T, metric='euclidean')
    distance, _ = kdt.query(pts_xformed_pred.T, k=1)
    mean_distance = np.mean(distance)
    print("mean_distance adds:",mean_distance)
    # Soglia e score binario
    threshold = diameter * percentage
    score = 1 if mean_distance < threshold else 0
    return score


def calc_score(pts3d, diameter, R_gt, t_gt, R_pred, t_pred, decay=0.05):
    if np.isnan(np.sum(t_pred)):
        return 0.0

    # Trasformazione dei punti
    pts_xformed_gt = R_gt @ pts3d.T + t_gt.reshape(3, 1)
    pts_xformed_pred = R_pred @ pts3d.T + t_pred.reshape(3, 1)

    # Distanza media punto più vicino (ADD-S)
    kdt = KDTree(pts_xformed_gt.T)
    distance, _ = kdt.query(pts_xformed_pred.T, k=1)
    mean_distance = np.mean(distance)

    # Score continuo tra 0 e 1, decresce con l'errore
    normalized_error = mean_distance / (diameter + 1e-6)
    score = np.exp(-normalized_error / decay)
    return score


def adi(R_est, t_est, R_gt, t_gt, pts,diameter,percentage=0.1):
    """Average Distance of Model Points for objects with indistinguishable
    views.

    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    #return e
    threshold = diameter * percentage
    print("diameter",diameter)
    print("mean distance:",e)
    print("threshol",threshold)
    # Score binario
    score = 1 if e < threshold else 0
    return score


def compute_model_diameter(model_points,obj_id,models_info_path):
    distances = cdist(model_points, model_points)  # matrice NxN delle distanze
    diameter = np.max(distances)
    return diameter

def rotation_error(R1, R2):
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    cos_theta = (trace - 1) / 2
    # Clamp il valore per evitare errori numerici fuori [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg






def save_pose_result(csv_path, scene_id, im_id, obj_id, score, R, t, time_taken):
    # Converti R (3x3) e t (3x1) in stringa
    R_str = ' '.join(['{:.6f}'.format(val) for val in R.flatten()])
    t_str = ' '.join(['{:.6f}'.format(val) for val in t.flatten()])

    # Crea la riga
    row = {
        'scene_id': scene_id,
        'im_id': im_id,
        'obj_id': obj_id,
        'score': round(score, 6),
        'R': R_str,
        't': t_str,
        'time': round(time_taken, 6)
    }

    # Scrittura (header solo se il file non esiste)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
















def compute_add_and_addS(folder, id_image,obj_id, pts3d, diameter, R_gt, t_gt, R_pred, t_pred, percentage=0.1):
    Add = compute_add_score_single(pts3d, diameter, R_gt, t_gt, R_pred, t_pred)    
    Add_S = compute_adds_score(pts3d, diameter, R_gt, t_gt, R_pred, t_pred)
    theta= rotation_error(R_gt,R_pred)
    delta = translation_error(t_gt,t_pred)
    result = {
        "id_image": id_image,
        "data": {
            "GT_R": R_gt.flatten().tolist(),
            "GT_T": t_gt.flatten().tolist(),
            "R_pred": R_pred.flatten().tolist(),
            "T_pred": t_pred.flatten().tolist(),
            "rotation_error": theta,
            "translation_error": delta,
            "ADD": Add,
            "ADD_S": Add_S
        }
    }
    obj_id_folder = str(int(obj_id)).zfill(6)
    score= calc_score(pts3d, diameter, R_gt, t_gt, R_pred, t_pred, decay=0.05)
    os.makedirs(f"/home/andrea/Desktop/Thesis_project/evaluation/{obj_id_folder}", exist_ok=True)

    json_path = f"/home/andrea/Desktop/Thesis_project/evaluation/{obj_id_folder}/results_{str(int(folder)).zfill(6)}.jsonl"
    # Scrivi in append, una riga = un oggetto JSON compatto
    #with open(json_path, "a") as f:
    #    f.write(json.dumps(result, separators=(',', ':')) + "\n")

    csv_path= f"/home/andrea/Desktop/Thesis_project/evaluation/csv/{obj_id_folder}/results_{str(int(folder)).zfill(6)}.csv"
    
    os.makedirs(f"/home/andrea/Desktop/Thesis_project/evaluation/csv/{obj_id_folder}", exist_ok=True)
    save_pose_result(csv_path=csv_path,
                     scene_id=folder,
                     im_id=id_image,
                     obj_id=obj_id,
                     time_taken=-1,
                     score=score,
                     R=R_pred,
                     t=t_pred)
    
    return Add, Add_S


def compute_add_and_addS_T(folder, id_image,obj_id, pts3d, diameter, R_gt, t_gt, R_pred, t_pred, percentage=0.1):
    #Add = compute_add_score_single(pts3d, diameter, R_gt, t_gt, R_pred, t_pred)  
    Add = add(pts3d,R_gt,t_gt,R_pred,t_pred,obj_id,diameter)  
    #Add_S = compute_adds_score(pts3d, diameter, R_gt, t_gt, R_pred, t_pred)
    Add_S = adi(R_pred,t_pred,R_gt,t_gt,pts3d,diameter)
    theta= rotation_error(R_gt,R_pred)
    delta = translation_error(t_gt,t_pred)
    result = {
        "id_image": id_image,
        "data": {
            "GT_R": R_gt.flatten().tolist(),
            "GT_T": t_gt.flatten().tolist(),
            "R_pred": R_pred.flatten().tolist(),
            "T_pred": t_pred.flatten().tolist(),
            "rotation_error": theta,
            "translation_error": delta,
            "ADD": Add,
            "ADD_S": Add_S
        }
    }
    obj_id_folder = str(int(obj_id)).zfill(6)
    
    os.makedirs(f"/home/andrea/Desktop/Thesis_project/evaluation/{obj_id_folder}", exist_ok=True)

    json_path = f"/home/andrea/Desktop/Thesis_project/evaluation/{obj_id_folder}/results_{str(int(folder)).zfill(6)}.jsonl"
    # Scrivi in append, una riga = un oggetto JSON compatto
    with open(json_path, "a") as f:
        f.write(json.dumps(result, separators=(',', ':')) + "\n")

    return Add, Add_S



def load_jsonl(filepath):
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def compute_precision(results, max_rot_deg=5, max_trans_cm=5):
    TP, FP = 0, 0
    for item in results:
        data = item["data"]
        rot_err = data["rotation_error"]
        trans_err = data["translation_error"] / 10.0  # mm → cm

        if rot_err <= max_rot_deg and trans_err <= max_trans_cm:
            TP += 1
        else:
            FP += 1

    total = TP + FP
    return (TP / total) * 100 if total > 0 else 0

def compute_ap_curve(results, mode="rotation", max_threshold=50, step=1):
    thresholds = list(range(1, max_threshold + 1, step))
    ap_values = []

    for t in thresholds:
        if mode == "rotation":
            ap = compute_precision(results, max_rot_deg=t, max_trans_cm=1000)
        elif mode == "translation":
            ap = compute_precision(results, max_rot_deg=360, max_trans_cm=t)
        ap_values.append(ap)

    return thresholds, ap_values

def plot_ap_curve_single_class(file_path, mode="rotation"):
    """
    Plotta la curva AP per una singola classe a partire da un file JSONL.

    :param file_path: Percorso al file .jsonl contenente i risultati.
    :param mode: "rotation" o "translation"
    """
    class_name = os.path.basename(file_path).replace(".jsonl", "")
    results = load_jsonl(file_path)

    thresholds, ap_values = compute_ap_curve(results, mode=mode)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, ap_values, marker='o', label=class_name)
    plt.xlabel("Rotation error (°)" if mode == "rotation" else "Translation error (cm)")
    plt.ylabel("AP (%)")
    plt.title(f"{mode.capitalize()} AP Curve - {class_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_add_percentage(results):
    add_scores,add_s_scores=0, 0
    total = 0
    for item in results:
        data = item["data"]
        add = data["ADD"]
        add_s = data["ADD_S"]
        total+=1
        if add == 1:
            add_scores += 1
        if add_s ==1:
            add_s_scores += 1
    
    #print("total",total)
    add_percentage = add_scores/total*100
    add_s_percentage = add_s_scores/total*100
    return add_percentage,add_s_percentage


def plot_all_jsonl_curves(folder_path, mode="rotation"):
    """
    Plotta un unico grafico con le curve AP di tutti i file JSONL nella cartella.
    """
    plt.figure(figsize=(10, 6))
    jsonl_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".jsonl"))

    for fname in jsonl_files:
        file_path = os.path.join(folder_path, fname)
        results = load_jsonl(file_path)
        thresholds, ap_values = compute_ap_curve(results, mode=mode)
        label = fname.replace(".jsonl", "")
        plt.plot(thresholds, ap_values, label=label)

    plt.xlabel("Rotation error (°)" if mode == "rotation" else "Translation error (cm)")
    plt.ylabel("AP (%)")
    plt.title(f"Combined {mode.capitalize()} AP Curves")
    plt.grid(True)
    plt.legend(title="Object")
    plt.tight_layout()
    plt.show()