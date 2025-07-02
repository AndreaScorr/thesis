import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
input_path1 =  "/home/andrea/Desktop/Thesis_project/Segmented/rgb/000053/000247_000013.png"
input_path2 =  "/home/andrea/Desktop/Thesis_project/Segmented/rgb/000053/000052_000013.png"

# Load an image
image = Image.open(input_path1)
output_path = "/home/andrea/Desktop/Thesis_project/Models/TRELLIS/"
file_name=input_path1.split("/")[-1].removesuffix(".png")
file_name=file_name.split("_")[-1]
output_path = output_path+file_name 
# Load an image
images = [
    Image.open(input_path1),
    Image.open(input_path2),
    #Image.open("assets/example_multi_image/character_3.png"),
]

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
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave(output_path+"sample_multi.mp4", video, fps=30)
outputs['gaussian'][0].save_ply(output_path +"sample.ply")
