'''import json

# Percorso al tuo file .jsonl
file_path = "/home/andrea/Desktop/Thesis_project/evaluation/000014/results_000055.jsonl"  # <-- Cambia con il percorso reale

# Leggi i dati dal file JSONL
data_list = []
with open(file_path, "r") as f:
    for line in f:
        parsed = json.loads(line)
        if "data" in parsed:
            data_list.append(parsed["data"])

# Calcolo del recall (per metrica binaria come ADD_S)
def compute_recall(data, key='ADD_S'):
    total = len(data)
    correct = sum(1 for d in data if d.get(key, 0) == 1)
    return correct / total if total > 0 else 0.0

# Calcola recall usando la metrica ADD_S
recall_add_s = compute_recall(data_list, key='ADD_S')

print(f"Average Recall (basato su ADD-S): {recall_add_s:.4f}")'''

import os
import json

def collect_jsonl_files(root_dir):
    jsonl_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".jsonl"):
                jsonl_paths.append(os.path.join(dirpath, f))
    return jsonl_paths

def compute_recall_from_file(jsonl_file, key='ADD_S'):
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line)["data"] for line in f if "data" in json.loads(line)]
    if not data:
        return 0.0, 0  # recall, count
    correct = sum(1 for d in data if d.get(key, 0) == 1)
    total = len(data)
    return correct / total, total

def compute_average_recall_across_files(root_dir, key='ADD_S'):
    files = collect_jsonl_files(root_dir)
    all_recalls = []
    weighted_correct = 0
    total_instances = 0

    for f in files:
        recall, count = compute_recall_from_file(f, key)
        all_recalls.append((f, recall))
        weighted_correct += recall * count
        total_instances += count

    avg_recall = weighted_correct / total_instances if total_instances else 0.0
    return avg_recall, all_recalls

# --------- USO ---------
root_directory = "/home/andrea/Desktop/Thesis_project/evaluation"  # <-- modifica qui
avg_recall, per_file_recalls = compute_average_recall_across_files(root_directory)

print(f"\n📊 Average Recall globale (ADD-S): {avg_recall:.4f}\n")

for path, r in per_file_recalls:
    print(f"{path}: recall = {r:.4f}")