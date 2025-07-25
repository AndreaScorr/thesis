import pandas as pd
import numpy as np

def compute_map(iou_value, thresholds=np.arange(0.5, 0.96, 0.05)):
    ap_list = [1 if iou_value >= t else 0 for t in thresholds]
    return sum(ap_list) / len(ap_list)

# Carica il CSV
df = pd.read_csv("/home/andrea/Desktop/Thesis_project/segmentation/evaluation/csv/IoU.csv")

# Filtra le scene da 48 a 59
df = df[(df['scene_id'] >= 48) & (df['scene_id'] <= 59)]

# Pulisce nomi colonne
df.columns = df.columns.str.strip()

# Conversione tipi
df["IoU"] = df["IoU"].astype(float)

# Calcola mAP per ogni riga
df["mAP"] = df["IoU"].apply(compute_map)

# Calcola media mAP per ogni oggetto
map_mean_per_obj = df.groupby("obj_id")["mAP"].mean().reset_index()

# Stampa e salva
print(map_mean_per_obj)
map_mean_per_obj.to_csv("/home/andrea/Desktop/Thesis_project/segmentation/evaluation/csv/mean_map_per_object.csv", index=False)
