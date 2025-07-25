import json
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
# Parte 1: Estrai i dati da tutti i file e salvali in un unico CSV
output_data = []

for i in range(1, 22):  # oggetti da 1 a 21
    for j in range(48, 60):  # scene da 48 a 59
        path = f"/home/andrea/Desktop/Thesis_project/evaluation_TRELLIS/{i}/results_{j:06}.jsonl"
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    id_image = obj.get("id_image", None)
                    data = obj.get("data", {})
                    rot_err = data.get("rotation_error", None)
                    trans_err = data.get("translation_error", None)

                    if id_image is not None and rot_err is not None and trans_err is not None:
                        output_data.append({
                            "object_id": i,
                            "scene_id": j,
                            "image_id": id_image,
                            "rotation_error": rot_err,
                            "translation_error": trans_err
                        })
        except Exception as e:
            print(f"Errore nel file {path}: {e}")
            continue

# Salva in CSV
csv_path = "/home/andrea/Desktop/Thesis_project/evaluation_TRELLIS/global_results.csv"
with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["object_id", "scene_id", "image_id", "rotation_error", "translation_error"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_data)

print(f"CSV salvato in: {csv_path}")

# Parte 2: Calcola precision curves
def compute_precision(results, max_rot_deg=5, max_trans_cm=5):
    TP, FP = 0, 0
    for item in results:
        rot_err = item["rotation_error"]
        trans_err = item["translation_error"] / 10.0  # mm → cm

        if rot_err <= max_rot_deg and trans_err <= max_trans_cm:
            TP += 1
        else:
            FP += 1

    total = TP + FP
    return (TP / total) * 100 if total > 0 else 0

def compute_ap_curve(results, mode="rotation", max_threshold=180, step=1):
    thresholds = list(range(1, max_threshold + 1, step))
    ap_values = []

    for t in thresholds:
        if mode == "rotation":
            ap = compute_precision(results, max_rot_deg=t, max_trans_cm=1000)
        elif mode == "translation":
            ap = compute_precision(results, max_rot_deg=360, max_trans_cm=t)
        ap_values.append(ap)

    return thresholds, ap_values


'''

csv_path1 = "/home/andrea/Desktop/Thesis_project/evaluation/global_results.csv"
csv_path2 = "/home/andrea/Desktop/Thesis_project/evaluation_TRELLIS/global_results.csv"


# Carica il CSV
df = pd.read_csv(csv_path)
results = df.to_dict(orient="records")

# Calcola AP curve
rot_thresholds, rot_ap = compute_ap_curve(results, mode="rotation", max_threshold=50, step=1)
trans_thresholds, trans_ap = compute_ap_curve(results, mode="translation", max_threshold=100, step=1)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rot_thresholds, rot_ap, label="Rotation Precision")
plt.xlabel("Rotation Threshold (degrees)")
plt.ylabel("Accuracy of Prediction (%)")
plt.title("AP Curve - Rotation")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(trans_thresholds, trans_ap, label="Translation Precision", color="orange")
plt.xlabel("Translation Threshold (cm)")
plt.ylabel("Accuracy of Prediction (%)")
plt.title("AP Curve - Translation")
plt.grid(True)

plt.tight_layout()
plt.show()'''

csv_path1 = "/home/andrea/Desktop/Thesis_project/evaluation/global_results.csv"
csv_path2 = "/home/andrea/Desktop/Thesis_project/evaluation_TRELLIS/global_results.csv"

# Funzioni già definite: compute_precision, compute_ap_curve

# Carica i dati da entrambi i CSV
df1 = pd.read_csv(csv_path1)
df2 = pd.read_csv(csv_path2)

results1 = df1.to_dict(orient="records")
results2 = df2.to_dict(orient="records")

# Calcola AP curves
rot_thresholds, rot_ap1 = compute_ap_curve(results1, mode="rotation", max_threshold=180, step=1)
_, rot_ap2 = compute_ap_curve(results2, mode="rotation", max_threshold=180, step=1)

trans_thresholds, trans_ap1 = compute_ap_curve(results1, mode="translation", max_threshold=80, step=1)
_, trans_ap2 = compute_ap_curve(results2, mode="translation", max_threshold=80, step=1)

# Plot comparativo
plt.figure(figsize=(12, 5))

# Grafico rotazione
plt.subplot(1, 2, 1)
plt.plot(rot_thresholds, rot_ap1, label="Ground Truth Model", linewidth=2)
plt.plot(rot_thresholds, rot_ap2, label="Trellis Model", linestyle="--", linewidth=2)
plt.xlabel("Rotation Threshold (degrees)")
plt.ylabel("Accuracy of Prediction (%)")
plt.title("AP Curve - Rotation")
plt.legend()
plt.grid(True)

# Grafico traslazione
plt.subplot(1, 2, 2)
plt.plot(trans_thresholds, trans_ap1, label="Ground Truth Model", linewidth=2)
plt.plot(trans_thresholds, trans_ap2, label="Trellis Model", linestyle="--", linewidth=2)
plt.xlabel("Translation Threshold (cm)")
plt.ylabel("Accuracy of Prediction (%)")
plt.title("AP Curve - Translation")
plt.legend()
plt.grid(True)

plt.tight_layout()
output_path="/home/andrea/Documents/thesis_material/trend"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

plt.show()