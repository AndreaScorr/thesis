import json
import numpy as np
def estrai_parametri(imgId,json_path, target_obj_id):
    """
    Estrae cam_R_m2c e cam_t_m2c per un dato obj_id da un file JSON.

    :param json_path: Percorso del file JSON.
    :param target_obj_id: L'obj_id di interesse.
    :return: Una tupla (cam_R_m2c, cam_t_m2c) oppure (None, None) se non trovato.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    items = data.get(str(imgId), [])

    for item in items:
        if item.get("obj_id") == target_obj_id:
            R = np.array(item["cam_R_m2c"]).reshape((3, 3))
            t = np.array(item["cam_t_m2c"])
            return R, t

    return None, None

'''
models_info_path="/home/andrea/Downloads/ycbv_train_pbr/train_pbr/000049/scene_gt.json"
gt_R,gt_T=estrai_parametri(imgId=547,json_path=models_info_path,target_obj_id=14)
print("GtR:",gt_R)
print("GtT:",gt_T)
'''