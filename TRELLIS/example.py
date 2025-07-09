import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import random
# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

input_path =  "/home/andrea/Desktop/Thesis_project/Segmented/rgb/000048/002070_000006.png"
# Load an image
image = Image.open(input_path)
output_path = "/home/andrea/Desktop/Thesis_project/Models/TRELLIS/"
file_name=input_path.split("/")[-1].removesuffix(".png")
file_name=file_name.split("_")[-1]
print(file_name) 

output_path=output_path+file_name+"/"
# Run the pipeline
outputs = pipeline.run(
    image,
    seed=2004995324, #random.randrange(1,2147483647),
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

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(output_path +"sample_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave(output_path +"sample_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave(output_path +"sample_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(output_path +"sample.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply(output_path +"sample.ply")
