import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils,postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
'''
input_path1 =  "TRELLIS/input/1/000024.png"
input_path2 =  "/home/andrea/Desktop/Thesis_project/TRELLIS/input/1/000168.png"

# Load an image
image = Image.open(input_path1)
output_path = "/home/andrea/Desktop/Thesis_project/TRELLIS/outputs/"
file_name=input_path2.split("/")[-2]
#file_name=file_name.split("_")[-1]
output_path = output_path+file_name 
# Load an image
images = [
    Image.open(input_path1),
    Image.open(input_path2),
    #Image.open("assets/example_multi_image/character_3.png"),
]'''
for i in range(1,22):
    input_dir = f"/home/andrea/Desktop/Thesis_project/TRELLIS/input/{i}/"
    output_root = "/home/andrea/Desktop/Thesis_project/TRELLIS/outputs/"

    # Estrai nome cartella finale (es. '1') per usarlo come nome della cartella output
    file_name = input_dir.rstrip("/").split("/")[-1]
    output_path = os.path.join(output_root, file_name)

    # Crea cartella output
    os.makedirs(output_path, exist_ok=True)

    # Ottieni e ordina le immagini PNG
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

    # Assumiamo di voler usare le prime due immagini
    input_path1 = os.path.join(input_dir, image_files[0])
    input_path2 = os.path.join(input_dir, image_files[1])

    # Carica immagini
    images = [Image.open(input_path1), Image.open(input_path2)]
    # Run the pipeline
    outputs = pipeline.run_multi_image(
        images,
        seed=1,
        # Optional parameters
        sparse_structure_sampler_params={
            "steps": 12,
            "cfg_strength": 7.5,
        },
        slat_sampler_params={
            "steps": 12,
            "cfg_strength": 3,
        },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes
    os.makedirs(output_path, exist_ok=True)

    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']

    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(output_path +"/"+"sample_gs.mp4", video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(output_path +"/"+"sample_rf.mp4", video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(output_path +"/"+"sample_mesh.mp4", video, fps=30)
    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(output_path +"/"+"sample.glb")

    #video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
    imageio.mimsave(output_path+"/"+"sample_multi.mp4", video, fps=30)
    outputs['gaussian'][0].save_ply(output_path +"/"+"sample.ply")
