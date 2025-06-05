import os
from collections import defaultdict

source_root = "/home/andrea/Desktop/Thesis_project/evaluation"
output_root = "/home/andrea/Desktop/Thesis_project/clastered_by_folder"

grouped = defaultdict(list)

# Scansiona i file per raggrupparli
for folder_name in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.startswith("results_") and filename.endswith(".jsonl"):
                result_id = filename.split(".")[0]  # es. "results_000048"
                grouped[result_id].append(folder_name)

# Crea la struttura di directory nel nuovo percorso
for result_id, folders in grouped.items():
    result_path = os.path.join(output_root, result_id)
    os.makedirs(result_path, exist_ok=True)
    for folder in folders:
        subfolder_path = os.path.join(result_path, folder)
        os.makedirs(subfolder_path, exist_ok=True)