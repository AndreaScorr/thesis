import os
import torch
import csv
from PIL import Image
from collections import defaultdict
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import json
import torch.nn as nn
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

with open("../eval/datasets/ycbv_test_bop19/test/id_to_label.json", "r") as f:
    id_to_label = json.load(f)

def print_sample_predictions(results, num_samples=5):
    print("\n--- Sample Prediction Results ---")
    for r in results[:num_samples]:
        print(f"Image           : {r['image']}")
        print(f"Ground Truth    : {r['ground_truth']}")
        print(f"Predicted Class : {r['predicted']}")
        print(f"Matched Image   : {os.path.basename(r['predicted_image'])}")
        print(f"Score           : {r['similarity_score']:.3f}")
        print()

def print_ap_per_class(ap_per_class):
    print("\n--- Average Precision per Class ---")
    for cls in sorted(ap_per_class):
        ap = ap_per_class[cls]
        ap_str = f"{ap:.4f}" if ap == ap else "NaN"
       # print(f"{cls}: {ap_str}")

def print_map_summary(map_score):
    print(f"\n--- Mean Average Precision (mAP) ---\nScore: {map_score:.4f}")

# Load object reference images and compute their features
def load_reference_image_embeddings(ref_dir):
    embeddings = []
    image_info = []

    for obj_folder in os.listdir(ref_dir):
        obj_path = os.path.join(ref_dir, obj_folder)
        if not os.path.isdir(obj_path):
           continue
       
        #image_embeddings = []
       
        for fname in os.listdir(obj_path):
            if fname.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(obj_path, fname)
                image = Image.open(img_path).convert("RGB")
                with torch.no_grad():
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state
                    features = features.mean(dim=1)  # Average pooling over the sequence length

                embeddings.append(features[0].cpu())  # Detach to CPU for efficiency
                image_info.append((obj_folder, img_path))  # Save object_id and full path

              # add the object id or name
    return embeddings, image_info

# Load bounding boxes
def load_boxes(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                object_id = parts[0]
                bbox = list(map(float, parts[1:]))
                boxes.append((object_id, bbox))
    return boxes

# Crop image
def crop_image(image, bbox):
    xmin, ymin, xmax, ymax = map(int, bbox)
    return image.crop((xmin, ymin, xmax, ymax))

# Evaluate a single image
def evaluate_image(image_path, box_path, ref_embeddings, ref_keys):
    results = []
    boxes = load_boxes(box_path)
    image = Image.open(image_path).convert("RGB")

    for object_id, bbox in boxes:
        cropped = crop_image(image, bbox)
        with torch.no_grad():
            inputs = processor(images=cropped, return_tensors="pt").to(device)
            outputs = model(**inputs)
            image_embed = outputs.last_hidden_state.mean(dim=1).cpu()

        similarities = cosine_similarity(image_embed.squeeze(0), ref_embeddings)
        print(f"Top 5 similarities: {sorted(similarities, reverse=True)[:5]}")

        best_idx = similarities.argmax()
        pred_id = ref_keys[best_idx]
        #print(pred_id)

        results.append({
            "image": os.path.basename(image_path),
            "ground_truth": object_id,
            "predicted": pred_id,
            "similarity_score": similarities[best_idx],
            "similarities": dict(zip(ref_keys, similarities.tolist()))
        })
    return results

# Evaluate dataset
def evaluate_bop_dataset(bop_root, ref_embeddings, ref_keys, id_to_label):
    all_results = []

    for scene_id in tqdm(os.listdir(bop_root), desc="Scenes"):
        scene_dir = os.path.join(bop_root, scene_id)
        if not os.path.isdir(scene_dir):
            continue

        scene_gt_path = os.path.join(scene_dir, 'scene_gt.json')
        scene_gt_info_path = os.path.join(scene_dir, 'scene_gt_info.json')
        rgb_dir = os.path.join(scene_dir, 'rgb')

        if not (os.path.exists(scene_gt_path) and os.path.exists(scene_gt_info_path) and os.path.isdir(rgb_dir)):
            continue

        with open(scene_gt_path, 'r') as f:
            scene_gt = json.load(f)
        with open(scene_gt_info_path, 'r') as f:
            scene_gt_info = json.load(f)

        for img_id_str in tqdm(scene_gt.keys(), desc=f"Images in {scene_id}", leave=False):
            img_id = int(img_id_str)
            image_path = os.path.join(rgb_dir, f"{img_id:06d}.png")
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            gt_instances = scene_gt[img_id_str]
            gt_infos = scene_gt_info[img_id_str]

            for obj_inst, info in zip(gt_instances, gt_infos):
                obj_id = obj_inst["obj_id"]
                object_label = id_to_label.get(str(obj_inst["obj_id"]))
                #object_label = id_to_label.get(str(obj_id))
                if object_label is None:
                    continue

                bbox = info.get("bbox_visib") or info.get("bbox_obj")
                if not bbox:
                    continue

                cropped = crop_image(image, [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                if cropped is None:
                    continue

                with torch.no_grad():
                    inputs = processor(images=cropped, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1).to(device)

                #print(cosine_similarity(image_embed, ref_embeddings))
                cos = nn.CosineSimilarity(dim=1)
                similarities = cos(features.squeeze(0), ref_embeddings)
                sim = (similarities+1)/2
                
                # Top 5 indices
                topk = torch.topk(similarities, k=5)
                top5_indices = topk.indices
                top5_scores = topk.values

                top5 = [(ref_keys[idx][0], ref_keys[idx][1], float(score)) for idx, score in zip(top5_indices, top5_scores)]

                best_idx = top5_indices[0]
                pred_id = ref_keys[best_idx][0]
                best_ref_path = ref_keys[best_idx][1]

                all_results.append({
                    "image": f"{scene_id}/{img_id:06d}.png",
                    "ground_truth": object_label,
                    "predicted": pred_id,
                    "predicted_image": best_ref_path,
                    "similarity_score": float(top5_scores[0]),
                    "top_5": top5,  # Save for further analysis
                    "similarities": {ref_keys[i][0]: float(similarities[i]) for i in range(len(ref_keys))}
                })
        # Debug snippet
    print("Sample prediction results:")
    for r in all_results[:5]:
        print(f"GT: {r['ground_truth']} | Predicted: {r['predicted']} | Score: {r['similarity_score']:.3f}")

    return all_results, ref_keys

# Save to CSV
def save_results_to_csv(results, filename="evaluation_results_dinov2.csv"):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["image", "ground_truth", "predicted", "similarity_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

# Accuracy
def print_accuracy(results):
    total = len(results)
    if total == 0:
        print("No results to evaluate.")
        return
    correct = sum(1 for r in results if r["ground_truth"] == r["predicted"])
    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct / total:.2%}")

def compute_map(results):
    y_true_dict = defaultdict(list)
    y_score_dict = defaultdict(list)
    label_set = set()
    ap5_list = []

    # Collect all labels
    for r in results:
        label_set.add(r["ground_truth"])
        label_set.update(r["similarities"].keys())

    # Fill y_true and y_score per label
    for r in results:
        gt = r["ground_truth"]
        similarities = r["similarities"]

        for label in label_set:
            y_true_dict[label].append(1 if gt == label else 0)
            y_score_dict[label].append(similarities.get(label, 0.0))

        # ---- mAP@5 ----
        top5 = r.get("top_5", [])  # Ensure it's available
        precisions = []
        correct = 0
        for i, pred_label in enumerate(top5[:5]):
            if pred_label[0] == gt:
                correct += 1
                precisions.append(correct / (i + 1))
        if precisions:
            ap5_list.append(sum(precisions) / len(precisions))
        else:
            ap5_list.append(0.0)

    # Calculate AP per label
    ap_per_label = {}
    for label in label_set:
        if sum(y_true_dict[label]) > 0:
            ap = average_precision_score(y_true_dict[label], y_score_dict[label])
            ap_per_label[label] = ap
        else:
            ap_per_label[label] = float('nan')

    # Final scores
    valid_aps = [v for v in ap_per_label.values() if not (v != v)]
    map_score = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0
    map5_score = sum(ap5_list) / len(ap5_list) if ap5_list else 0.0

    # Print
    print("\nAverage Precision per Label (mAP):")
    for label, ap in ap_per_label.items():
        print(f"{label}: {ap:.4f}" if ap == ap else f"{label}: NaN")
    
    print(f"\nMean Average Precision (mAP): {map_score:.4f}")
    print(f"Mean Average Precision @5 (mAP@5): {map5_score:.4f}")

# Main
if __name__ == "__main__":
    ref_dir = "../object_images/"
    bop_data_root = "../eval/datasets/ycbv_test_bop19/test"

    ref_embeddings, ref_keys = load_reference_image_embeddings(ref_dir)
    ref_embeddings = torch.stack(ref_embeddings).to(device)

    results, all_classes = evaluate_bop_dataset(bop_data_root, ref_embeddings, ref_keys, id_to_label)

    save_results_to_csv(results, "evaluation_results_dinov2.csv")
    print_accuracy(results)
   # print(all_classes)
    compute_map(results)
