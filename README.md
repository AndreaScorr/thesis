# Thesis Project #
Estimating the 6D pose of objects is a fundamental task in robotics,
AR/VR, and autonomous systems, enabling semantic manipulation
and interaction with real-world environments. Traditional pipelines
rely on annotated datasets, CAD models, or category-specific training, which are computationally expensive and difficult to scale to
novel objects. This thesis investigates a zero-shot pipeline for 6D
object pose estimation that combines language-guided segmentation
and novel view synthesis, aiming to reduce dependency on handcrafted
3D models. The proposed pipeline takes as input an image, a textual
prompt, which could be provided as a user instruction, and a few reference views of the target object. As output, it estimates the object’s
6D pose, represented by three variables for rotation (angles) and three
for translation in the camera coordinate system. Specifically, segmentation is performed using Grounded-SAM, while 3D reconstruction is
achieved with TRELLIS, a structured latent model capable of generating object model from a few input views. The approach is evaluated
on the YCB-V dataset, benchmarking the performance of TRELLISgenerated models against ground-truth CAD models. Experimental
results show that the pipeline achieves competitive performance using ground-truth CAD models, but novel view synthesis fall short in
pose estimation accuracy. The limitations come from inaccuracies in
shape reconstruction, difficulties in reproducing detailed textures, and
sparsity of feature correspondences.These results motivate further research into improving generative 3D modeling methods, to make them
viable alternatives to CAD models in pose estimation pipelines.

# Project outline #
The following scheme illustrates the overall workflow of the proposed method.  
<img width="1661" height="2338" alt="pipeline_2" src="https://github.com/user-attachments/assets/ca26ff29-6881-4785-a80c-12b1a73a26d3" />
## 🧩 Proposed Approach Overview

The process consists of several main phases:

- **Language-Guided Object Segmentation**  
  A textual prompt is provided to **GroundedSAM** [(Li et al., 2022)](https://arxiv.org/abs/2305.04360), which combines the open-set object detector **Grounding DINO** [(Liu et al., 2024)](https://arxiv.org/abs/2303.05499) with the **Segment Anything Model (SAM)** [(Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643) to generate a mask of the queried object.

- **Template Rendering**  
  Around **200 templates** are rendered from a 3D model (either ground truth or reconstructed).  
  Using **Normalized Object Coordinate Space (NOCS)** [(Wang et al., 2019)](https://arxiv.org/abs/1901.02970), objects are normalized into a unit cube and mapped to RGB values for visualization.

- **3D Reconstruction**  
  To simulate different viewpoints observed by a robot, a set of template images is provided as input to **TRELLIS**, a novel view synthesis method based on **Structured Latent Models** [(Xiang et al., 2025)].  
  TRELLIS reconstructs a 3D representation of the object.

- **Template Matching**  
  Patch features are extracted from both the input image and the rendered templates.  
  A **weighted combination of patchwise similarity and SIFT score** [(Lowe, 2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) is used to select the most fitting template.

- **Pose Estimation**  
  Visual descriptors are extracted using **pre-trained Vision Transformers (ViT)** [(Caron et al., 2021)](https://arxiv.org/abs/2011.12247).  
  The descriptors match templates with query images, producing geometric correspondences used to estimate the object’s **6D pose** via **RANSAC / PnP**.

---

📘 *References correspond to the main components utilized in the pipeline.*
