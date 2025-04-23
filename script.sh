set -e
#CONFIG_PATH="/home/andrea/Desktop/Thesis_project/scrissor.yaml"
#CONFIG_PATH="/home/andrea/Desktop/Thesis_project/config.yaml"
CONFIG_PATH="/home/andrea/Desktop/Thesis_project/mug.yaml"
#CONFIG_PATH="/home/andrea/Desktop/Thesis_project/box.yaml"

python /home/andrea/Desktop/Thesis_project/segmentation/segmentation_function.py --config "$CONFIG_PATH"
python /home/andrea/Desktop/Thesis_project/TRELLIS/modelGenerator.py --config "$CONFIG_PATH"
blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render_nocs.py --config "$CONFIG_PATH"
blenderproc run /home/andrea/Desktop/Thesis_project/Nocs/render.py --config "$CONFIG_PATH"
python /home/andrea/Desktop/Thesis_project/FeatureMatching/featureMatching.py --config "$CONFIG_PATH"