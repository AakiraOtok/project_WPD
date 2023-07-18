from utils.lib import *
from utils.VOC_utils import VOCUtils
from model.SSD300 import SSD300
from model.SSD512 import SSD512
from pycocotools.coco import COCO
from utils.COCO_utils import COCOUtils
from model.FPN_SSD300 import FPN_SSD300

#ann_file = r"H:\data\COCO\instances_valminusminival2014.json"
#coco = COCO(annotation_file=ann_file)
#class_dict = coco.cats
#idx = list(class_dict.keys())
#content_dict  = list(class_dict.values())
#name = []
#for content in content_dict:
    #name.append(content["name"])

#ori_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
#idx     = list(range(len(ori_idx)))
#name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#ori_idx2idx = dict(zip(ori_idx, idx))
#idx2name = dict(zip(idx, name))
#print(ori_idx2idx)

#dataset = COCOUtils(r"H:\data\COCO\val2014", r"H:\data\COCO\instances_minival2014.json").make_dataset(phase="valid")
#print(len(dataset))

#T = SSD512()
#print(T.create_prior_boxes().shape)

#a = torch.ones(1, 1024, 19, 19)
#convtp  = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2)
#conv1x1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
#b = convtp(a)
#b = conv1x1(b)

#print(b.shape)

#a = torch.FloatTensor([5])
#b = torch.FloatTensor(5)
#print(torch.pow(b, a))

model = FPN_SSD300()
model = SSD300()
t = sum(p.numel() for p in model.parameters())
print(t)