from utils.lib import *
from utils.VOC_utils import VOCUtils, class_inverse_map
data_folder_path = r"H:\projectWPD\data"
voc = VOCUtils(data_folder_path)

state_dict = torch.load(r"H:\projectWPD\checkpoint\ssd300_mAP_77.43_v2.pth")
for param in state_dict.keys():
    print(param)