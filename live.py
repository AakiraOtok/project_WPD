from utils.lib import *
from model.SSD300 import SSD300
from model.SSD512 import SSD512
from ssd import build_ssd
from utils.box_utils import draw_bounding_box, Non_Maximum_Suppression
from utils.VOC_utils import VOC_name2idx, VOC_idx2name
from utils.augmentations_utils import CustomAugmentation
from collections import deque

def create_prior_boxes():
    """ 
    Tạo 8732 prior boxes (tensor) như trong paper
    mỗi box có dạng [cx, cy, w, h] được scale
    """
    # kích thước feature map tương ứng
    fmap_sizes    = [38, 19, 10, 5, 3, 1]
        
    # scale như trong paper và được tính sẵn thay vì công thức
    # lưu ý ở conv4_3, tác giả xét như một trường hợp đặc biệt (scale 0.1):
    # Ở mục 3.1, trang 7 : 
    # "We set default box with scale 0.1 on conv4 3 .... "
    # "For SSD512 model, we add extra conv12 2 for prediction, set smin to 0.15, and 0.07 on conv4 3...""
    box_scales    = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]

            
    aspect_ratios = [
            [1., 2., 0.5],
            [1., 2., 3., 0.5, 0.333],
            [1., 2., 3., 0.5, 0.333],
            [1., 2., 3., 0.5, 0.333],
            [1., 2., 0.5],
            [1., 2., 0.5]
        ]
    dboxes = []
        
        
    for idx, fmap_size in enumerate(fmap_sizes):
        for i in range(fmap_size):
            for j in range(fmap_size):

                # lưu ý, cx trong ảnh là trục hoành, do đó j + 0.5 chứ không phải i + 0.5
                cx = (j + 0.5) / fmap_size
                cy = (i + 0.5) / fmap_size

                for aspect_ratio in aspect_ratios[idx]:
                    scale = box_scales[idx]
                    dboxes.append([cx, cy, scale*sqrt(aspect_ratio), scale/sqrt(aspect_ratio)])

                    if aspect_ratio == 1:
                        try:
                            scale = sqrt(scale*box_scales[idx + 1])
                        except IndexError:
                            scale = 1.
                        dboxes.append([cx, cy, scale*sqrt(aspect_ratio), scale/sqrt(aspect_ratio)])

    dboxes = torch.FloatTensor(dboxes)
        
    #dboxes = pascalVOC_style(dboxes)
    dboxes.clamp_(min=0, max=1)
     #dboxes = yolo_style(dboxes)
                
    return dboxes

def live_cam(model, cam, size=300, num_classes=21, mapping = VOC_idx2name):
    model.to("cuda")
    dboxes = create_prior_boxes().to("cuda")
    d = deque()

    while cv2.waitKey(1) != ord('q'):
        ret, img = cam.read()
        if not ret:
            break

        aug = CustomAugmentation(size=size)
    
        transformed_img, _1, _2, _3 = aug(img, phase="valid")
        transformed_img        = torch.FloatTensor(transformed_img[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous().to("cuda")

        offset, conf = model(transformed_img.unsqueeze(0))
        
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.2, iou_threshold=0.45, top_k=200, num_classes=num_classes)
        draw_bounding_box(img, pred_bboxes, pred_labels, pred_confs, mapping)
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
    pretrain_path = r"H:\project_WPD\iteration_30000.pth"
    n_classes     = 21

    #model = SSD300(pretrain_path, n_classes=n_classes)
    #model = SSD512(pretrain_path, n_classes=n_classes)
    model = build_ssd('test', 300, 21)
    model.eval()

    cam   = cv2.VideoCapture(0)
    live_cam(model, cam, size=512)
