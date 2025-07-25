import os
import csv
import numpy as np  # per media ignorando NaN
import Evaluation_utils as Eval
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# Dizionario oggetti
import os
import csv
import numpy as np

# Dizionario oggetti
oggetti = {
    1: "master_chef_can",
    2: "cracker_box",
    3: "sugar_box",
    4: "tomato_soup_can",
    5: "mustard_bottle",
    6: "tuna_fish_can",
    7: "pudding_box",
    8: "gelatin_box",
    9: "potted_meat_can",
    10: "banana",
    11: "pitcher_base",
    12: "bleach_cleanser",
    13: "bowl",
    14: "mug",
    15: "power_drill",
    16: "wood_block",
    17: "scissor",
    18: "large_marker",
    19: "large_clamp",
    20: "extra_large_clamp",
    21: "foam_brick"
}

# Parametri
base_folder = "evaluation_TRELLIS"
scene_range = range(48, 60)
output_csv = "mean_add_adds_by_object_trellis.csv"

# Header CSV
header = ["object_name", "mean_add (%)", "mean_add-s (%)"]

# Righe da scrivere
rows = []

for obj_id in range(1, 22):
    obj_name = oggetti[obj_id]
    add_values = []
    adds_values = []

    for scene_id in scene_range:
        file_path = os.path.join(base_folder, str(obj_id), f"results_{scene_id:06d}.jsonl")
        if os.path.exists(file_path):
            try:
                results = Eval.load_jsonl(file_path)
                add, adds = Eval.compute_add_percentage(results)
            except Exception as e:
                print(f"Errore nel file {file_path}: {e}")
                add, adds = np.nan, np.nan
        else:
            print(f"File non trovato: {file_path}")
            add, adds = np.nan, np.nan

        add_values.append(add)
        adds_values.append(adds)

    mean_add = np.nanmean(add_values)
    mean_adds = np.nanmean(adds_values)

    row = [
        obj_name,
        f"{mean_add:.2f}" if not np.isnan(mean_add) else "NaN",
        f"{mean_adds:.2f}" if not np.isnan(mean_adds) else "NaN"
    ]
    rows.append(row)

# Scrivi il CSV
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nMedia ADD/ADD-S salvata in: {output_csv}")
