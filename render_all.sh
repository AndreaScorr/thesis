set -e

blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render_nocs.py --config obj_config/16_wood_block.yaml
blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render.py --config obj_config/16_wood_block.yaml

blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render_nocs.py --config obj_config/17_scrissor.yaml
blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render.py --config obj_config/17_scrissor.yaml

blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render_nocs.py --config obj_config/18_large_marker.yaml
blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render.py --config obj_config/18_large_marker.yaml

blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render_nocs.py --config obj_config/19_large_clamp.yaml
blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render.py --config obj_config/19_large_clamp.yaml

blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render_nocs.py --config obj_config/20_extra_large_clamp.yaml
blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render.py --config obj_config/20_extra_large_clamp.yaml

blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render_nocs.py --config obj_config/21_foam_brick.yaml
blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render.py --config obj_config/21_foam_brick.yaml