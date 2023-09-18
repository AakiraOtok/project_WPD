from torchvision import transforms 
from model.FPN_SSD300_b import FPN_SSD300
from utils.box_utils import Non_Maximum_Suppression, coco_style, draw_bounding_box, yolo_style
from utils.COCO_utils import ori_idx2handle_idx, handle_idx2ori_idx, COCO_idx2name
import json
from PIL import Image
import cv2
import os
import numpy as np
from tqdm import tqdm 

def get_img_dict(ann_path):
    with open(ann_path) as f:
        data = json.load(f)
    return data['images']

if __name__ == "__main__":
    data_folder_path = r"H:\data\COCO\test2017"
    ann_path         = r"H:\data\COCO\image_info_test-dev2017.json"
    pretrain_path    = r"H:\projectWPD\VOC_checkpoint\iteration_900000.pth"
    result_file      = r"H:\checkpoint\result.json"
    num_classes      = 81

    images_dict = get_img_dict(ann_path)

    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    model = FPN_SSD300(pretrain_path, data_train_on="COCO", n_classes=num_classes)
    model.to('cuda')

    dboxes = model.create_prior_boxes().to("cuda")
    comma = False

    with open(result_file, 'w') as f:
        f.write("[")
        for img_dict in tqdm(images_dict):
            id = img_dict['id']
            w  = img_dict['width']
            h  = img_dict['height']
            file_name = img_dict['file_name']

            original_image = cv2.imread(os.path.join(data_folder_path, file_name))
            image = Image.fromarray(original_image)
            image          = normalize(to_tensor(resize(image)))

            image          = image.unsqueeze(0).to('cuda')
            offset, conf = model(image)
            pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.01, iou_threshold=0.45, top_k=200, num_classes=num_classes)
            if pred_bboxes is not None:
                #draw_bounding_box(original_image, pred_bboxes, pred_labels, pred_confs, COCO_idx2name)
                #pred_bboxes = coco_style(pred_bboxes)
                pred_bboxes = yolo_style(pred_bboxes)
                for box, label, conf in zip(pred_bboxes, pred_labels, pred_confs):
                    sub_dict = {}
                    sub_dict['image_id'] = id
                    sub_dict['category_id'] = handle_idx2ori_idx[label.item()]
                    sub_dict['bbox'] = [(box[0]*w).item(), (box[1]*h).item(), (box[2]*w).item(), (box[3]*h).item()]
                    sub_dict['score'] = conf.item()
                    if comma:
                        f.write(', ')
                    else:
                        comma = True
                    json.dump(sub_dict, f)
                #cv2.imshow('img', original_image)
                #cv2.waitKey()
        f.write("]")