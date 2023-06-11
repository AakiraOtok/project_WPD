from lib import *
from model import SSD
from VOC_utils import VOCUtils, collate_fn, class_inverse_map
from box_utils import Non_Maximum_Suppression

def detect(dataloader, model):
    model.to("cuda")
    for images, bboxes, labels, difficulties in dataloader:

        images = images.to("cuda")
        offset, conf, dboxes = model(images)
        offset = offset.to("cuda")
        conf   = conf.to("cuda")
        dboxes = dboxes.to("cuda")
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.5, iou_threshold=0.45, top_k=200)

        img = images.squeeze(0).permute(1, 2, 0).contiguous()[:, :, (2, 1, 0)].cpu().numpy()
        bboxes = pred_bboxes
        labels = pred_labels
        
        H, W, C = img.shape
        H -= 1
        W -= 1

        if bboxes is not None:
            for box, label in zip(bboxes, labels):
                p1 = (int(box[0]*W), int(box[1]*H))
                p2 = (int(box[2]*W), int(box[3]*H))
                cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
                cv2.putText(img, class_inverse_map[label.item()], p1, 1, 1, (0, 255, 0), 1)
        cv2.imshow("img", img)
        cv2.waitKey()


if __name__ == "__main__":
    data_folder_path = r"H:\projectWPD\data"
    dataloader = VOCUtils(data_folder_path).make_dataloader(r"VOC2007", r"test.txt", 1, True, collate_fn, phase='valid',num_worker=4, pin_memory=True)
    model      = SSD(pretrain_path=r"H:\projectWPD\VOC2007_trainval_checkpoint\iteration_6000.pth", n_classes=21)
    detect(dataloader, model)