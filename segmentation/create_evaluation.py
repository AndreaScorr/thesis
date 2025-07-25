import pandas as pd

# Carica il CSV
df = pd.read_csv("/home/andrea/Desktop/Thesis_project/segmentation/evaluation/csv/IoU.csv")

# Filtra le scene da 48 a 59
df = df[(df['scene_id'] >= 48) & (df['scene_id'] <= 59)]

# Rimuovi spazi bianchi dai nomi delle colonne
df.columns = df.columns.str.strip()

# Assicuriamoci che IoU sia float
df["IoU"] = df["IoU"].astype(float)

# Calcolo della media per ogni oggetto (senza scena)
iou_mean_per_obj = df.groupby("obj_id")["IoU"].mean().reset_index()

# Stampa o salva su file
print(iou_mean_per_obj)
iou_mean_per_obj.to_csv("mean_iou_per_object.csv", index=False)
