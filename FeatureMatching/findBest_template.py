'''import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from torch.nn.functional import cosine_similarity

# === CONFIG ===
TEMPLATE_FOLDER = "blender_render/obj_000019.ply"  # <- Cambia con il path della tua cartella
QUERY_IMAGE = "/home/andrea/Desktop/Thesis_project/FeatureMatching/buffer/crop.png"      # <- Cambia con il path dell'immagine da confrontare
MODEL_NAME = "facebook/dinov2-large"    # Puoi usare anche dinov2-large o dinov2-gigantic

# === CARICA MODELLO ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

def find_Best_template(query_path,template_folder):
    # === FUNZIONE PER ESTRARRE EMBEDDING ===
    def get_embedding(image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)  # [768]

    # === EMBEDDING IMMAGINE DI QUERY ===
    print("Estrazione embedding dell'immagine da confrontare...")

    query_embedding = get_embedding(query_path)
    #query_embedding = get_embedding(QUERY_IMAGE)

    # === LOOP SUI TEMPLATE ===
    print("Confronto con i template...")
    best_score = -1
    best_template = None
    similarities = []

    #for filename in tqdm(sorted(os.listdir(TEMPLATE_FOLDER))):
    for filename in tqdm(sorted(os.listdir(template_folder))):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        template_path = os.path.join(template_folder, filename)
        template_embedding = get_embedding(template_path)
        similarity = cosine_similarity(query_embedding, template_embedding, dim=0).item()
        similarities.append((filename, similarity))
        if similarity > best_score:
            best_score = similarity
            best_template = filename

    # === RISULTATI ===
    print("\n🔍 Miglior corrispondenza trovata:")
    print(f"Template: {best_template}")
    print(f"Similarità: {best_score:.4f}")
# (Opzionale) Stampa top 5 risultati
    similarities.sort(key=lambda x: x[1], reverse=True)
    print("\n🏆 Top 5 template più simili:")
    for i in range(min(5, len(similarities))):
        print(f"{i+1}. {similarities[i][0]} -> Similarità: {similarities[i][1]:.4f}")
    return best_template

    
    
find_Best_template(query_path=QUERY_IMAGE,template_folder=TEMPLATE_FOLDER)

import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#ref_dir = "blender_render/obj_000019.ply"  # <- Cambia con il path della tua cartella
#input_image_path = "/home/andrea/Desktop/Thesis_project/FeatureMatching/buffer/crop.png"      # <- Cambia con il path dell'immagine da confrontare

# Load model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

# Step 1: Load reference templates
def load_reference_image_embeddings(ref_dir):
    embeddings = []
    image_paths = []
    for fname in tqdm(os.listdir(ref_dir), desc="Loading templates"):
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(ref_dir, fname)
            image = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(features.squeeze(0).cpu())
            image_paths.append(img_path)
    return torch.stack(embeddings), image_paths

# Step 2: Embed input image
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        #features = outputs.last_hidden_state.mean(dim=1)
        features = outputs.last_hidden_state[:, 1:, :]  # Rimuove il token CLS, tiene solo le patch
        features = torch.nn.functional.normalize(features, dim=-1)  # Normalizza le patch
    return features.squeeze(0).cpu()

def patchwise_similarity(patch_feats_input, patch_feats_template):
    # input: [num_patches, dim]
    # template: [num_patches, dim]
    sim_matrix = cosine_similarity(patch_feats_input.numpy(), patch_feats_template.numpy())
    return sim_matrix.max()  # oppure: sim_matrix.mean()

# Step 3: Compute similarity
def find_most_similar(input_embedding, template_embeddings, image_paths, top_k=5):
    similarities = cosine_similarity(input_embedding.unsqueeze(0), template_embeddings)[0]
    top_k_idx = similarities.argsort()[-top_k:][::-1]
    return [(image_paths[i], similarities[i]) for i in top_k_idx]



def find_Best_template(input_image_path,ref_dir):

    print("Loading template embeddings...")
    template_embeddings, image_paths = load_reference_image_embeddings(ref_dir)
    
    print("Embedding input image...")
    input_embedding = embed_image(input_image_path)

    print("Finding most similar images...")
    results = find_most_similar(input_embedding, template_embeddings, image_paths, top_k=5)

    print("\nTop matches:")
    for path, score in results:
        print(f"{os.path.basename(path)} | Similarity: {score:.4f}")

    return (os.path.basename(results[2][0]))

#best=find_Best_template(input_image_path,ref_dir)
#print("best",best)'''

import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/dinov2-large"

# === MODELLO ===
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# === FUNZIONI ===

def extract_patch_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # Esclude CLS
        patch_tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
    return patch_tokens.squeeze(0).cpu()  # [num_patches, dim]

def compute_patchwise_similarity(patch_feats_input, patch_feats_template):
    sim_matrix = cosine_similarity(patch_feats_input.numpy(), patch_feats_template.numpy())
    return sim_matrix.max(axis=1).mean()  # media dei best match

def load_all_template_embeddings(template_dir):
    template_embeddings = []
    image_paths = []

    for fname in tqdm(sorted(os.listdir(template_dir)), desc="Caricamento template"):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")) and not fname.endswith("_uv.png"):
            img_path = os.path.join(template_dir, fname)
            feats = extract_patch_features(img_path)
            template_embeddings.append(feats)
            image_paths.append(img_path)


    return template_embeddings, image_paths
def compute_sift_score(query_path, template_path):
    # Load images in grayscale
    img1 = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0  # No features found
    
    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe’s ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    angles = []
    for m in good:
        dx1 = kp1[m.queryIdx].angle
        dx2 = kp2[m.trainIdx].angle
        angles.append(abs(dx1 - dx2))

    mean_angle = np.mean(angles)
    if mean_angle > 90:  # soglia tunabile
        return 0  # Penalizza rotazioni troppo forti
    
    if len(good) < 6:
        return 0  # Not enough matches for homography

    # Estimate homography with RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is not None:
        det = np.linalg.det(H[:2, :2])
        if det < 0:  # riflessione (flipping)
            return 0  # penalizza match speculari
    if mask is None:
        return 0

    inliers = mask.sum()
    return inliers / len(good)  # Ratio of good geometric matches

def find_Best_template_patchwise(input_image_path, template_dir, top_k=5, alpha=0.7, beta=0.3):
    print("\n📦 Estrai embedding immagine di input...")
    input_feats = extract_patch_features(input_image_path)

    print("📁 Caricamento e confronto con i template...")
    template_feats_list, template_paths = load_all_template_embeddings(template_dir)

    print("\n🔍 Calcolo similarità patch-wise DINOv2...")
    dino_scores = []
    for feats, path in zip(template_feats_list, template_paths):
        sim = compute_patchwise_similarity(input_feats, feats)
        dino_scores.append((path, sim))

    dino_scores.sort(key=lambda x: x[1], reverse=True)
    top_templates = dino_scores[:top_k]

    print("\n🔄 Reranking con SIFT + RANSAC...")
    final_ranked = []
    for path, dino_score in top_templates:
        sift_score = compute_sift_score(input_image_path, path)
        combined_score = alpha * dino_score + beta * sift_score
        final_ranked.append((path, combined_score))

    final_ranked.sort(key=lambda x: x[1], reverse=True)

    print("\n🏁 Ranking finale:")
    for i, (path, score) in enumerate(final_ranked):
        print(f"{i+1}. {os.path.basename(path)} | Score combinato: {score:.4f}")

    return os.path.basename(final_ranked[0][0])