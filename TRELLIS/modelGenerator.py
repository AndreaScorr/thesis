import os
import yaml
import argparse
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config_path, updates):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config.update(updates)

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
args = parser.parse_args()

config = load_config(args.config)
text_prompt= config["text_prompt"]
origin_path=config["image_path"].split("/")[-1]
image_path=config["segmentation_path"] + "/" + text_prompt + "_" +origin_path
#print(image_path)


# Load an image
image = Image.open(image_path)

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

output_path = config["segmentation_path"].removesuffix("Segmented/rgb")
output_path = output_path+"Models/"

file_name = text_prompt+"_"+origin_path.split(".")[0]
print(file_name) 

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(output_path+file_name+"_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave(output_path+file_name+"_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave(output_path+file_name+"_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb_path=output_path+file_name+"_sample.glb"
glb.export(glb_path)

ply_path= output_path+file_name+"_sample.ply"
# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply(ply_path)

update_config(args.config, 
 {"glb_path": glb_path,
  "ply_path":ply_path,
  "model_path":output_path
    }             
)
