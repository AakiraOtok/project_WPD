from utils.lib import *
from model import SSD
from utils.box_utils import draw_bounding_box, Non_Maximum_Suppression
from utils.VOC_utils import class_direct_map
from utils.augmentations_utils import CustomAugmentation

def live_cam(model, cam):
    model.to("cuda")
    dboxes = model.create_prioir_boxes().to("cuda")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        aug = CustomAugmentation(phase="valid")
    
        transformed_img, _1, _2, _3 = aug(img)
        transformed_img        = torch.FloatTensor(transformed_img[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous()
        offset, conf = model(transformed_img.unsqueeze(0))
        
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.2, iou_threshold=0.45, top_k=200)
        draw_bounding_box(img, pred_bboxes, pred_labels, pred_confs, class_direct_map)
        cv2.imshow("img", img)
        cv2.waitKey()





if __name__ == "__main__":
    pretrain_path = ""
    model = SSD(pretrain_path, n_classes=21)
    cam   = cv2.VideoCapture(0)

    live_cam(model, cam)
