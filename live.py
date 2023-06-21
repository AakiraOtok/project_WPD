from utils.lib import *
from model.SSD300 import SSD
from utils.box_utils import draw_bounding_box, Non_Maximum_Suppression
from utils.VOC_utils import class_direct_map, class_inverse_map
from utils.augmentations_utils import CustomAugmentation
from collections import deque

def live_cam(model, cam):
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    d = deque()

    while cv2.waitKey(1) != ord('q'):
        ret, img = cam.read()
        if not ret:
            break

        aug = CustomAugmentation()
    
        transformed_img, _1, _2, _3 = aug(img, phase="valid")
        transformed_img        = torch.FloatTensor(transformed_img[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous().to("cuda")

        offset, conf = model(transformed_img.unsqueeze(0))
        
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.2, iou_threshold=0.45, top_k=200)
        draw_bounding_box(img, pred_bboxes, pred_labels, pred_confs, class_inverse_map)
        H, W, C = img.shape
        
        t = time.time()
        d.append(t)
        if len(d) > 1000:
            d.popleft()

        if len(d) >= 100:
            fps = str(int(len(d)/(d[-1] - d[0])))
        else:
            fps = "Measuring" 
        cv2.putText(img, "FPS : " + fps, (int(W - W*20/100), int(H*5/100)), 1, 1, (0, 255, 0), 1)
        cv2.imshow("img", img)

if __name__ == "__main__":
    pretrain_path = r"H:\projectWPD\checkpoint\iteration_120000.pth"
    model = SSD(pretrain_path, n_classes=21)
    cam   = cv2.VideoCapture(0)

    live_cam(model, cam)
