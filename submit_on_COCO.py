from utils.lib import *
from torchvision import transforms
from model.FPN_SSD300_b import FPN_SSD300_b
from utils.box_utils import Non_Maximum_Suppression, yolo_style
from utils.COCO_utils import ori_idx2handle_idx, handle_idx2ori_idx
import json

def get_img_dict(ann_path):
    with open(ann_path) as f:
        content = f.read()

    data = json.load(content)
    return data['images']

if __name__ == "__main__":
    data_folder_path = r''
    ann_path         = r''
    pretrain_path    = r''
    result_file      = r''
    num_classes      = 81

    images_dict = get_img_dict(ann_path)
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    model = FPN_SSD300_b(pretrain_path, data_train_on="COCO", n_classes=num_classes)
    model.to('cuda')

    dboxes = model.create_prior_boxes().to("cuda")
    result_list = []

    for img_dict in images_dict:
        id = img_dict['id']
        w  = img_dict['width']
        h  = img_dict['height']
        file_name = img_dict['file_name']

        original_image = cv2.imread(os.path.join(data_folder_path, file_name))
        image          = normalize(to_tensor(resize(original_image)))
        image          = image.unsqueeze(0).to('cuda')
        offset, conf = model(image)
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.3, iou_threshold=0.45, top_k=200, num_classes=num_classes)
        pred_bboxes = yolo_style(pred_bboxes)
        if pred_bboxes is not None:
            for box, label, conf in zip(pred_bboxes, pred_labels, pred_confs):
                sub_dict = {}
                sub_dict['image_id'] = id
                sub_dict['category_id'] = handle_idx2ori_idx[label.item()]
                sub_dict['bbox'] = [int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)]
                sub_dict['score'] = conf.item()
                result_list += [sub_dict]
    
    json_data = json.dumps(data)

    #   Ghi chuỗi JSON vào file
    with open(result_file, 'w') as file:
        file.write(json_data)

    
