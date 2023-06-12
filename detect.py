from lib import *
from model import SSD
from VOC_utils import VOCUtils, collate_fn, class_inverse_map
from box_utils import Non_Maximum_Suppression

def detect(dataset, model):
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    #for images, bboxes, labels, difficulties in dataloader:
    for idx in range(dataset.__len__()):
        origin_image, images, bboxes, labels, difficulties = dataset.__getitem__(idx, get_origin_image=True)

        images = images.unsqueeze(0).to("cuda")
        offset, conf = model(images)
        offset = offset.to("cuda")
        conf   = conf.to("cuda")
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.5, iou_threshold=0.45, top_k=200)

        bboxes = pred_bboxes
        labels = pred_labels
        
        H, W, C = origin_image.shape
        H -= 1
        W -= 1

        if bboxes is not None:
            for box, label in zip(bboxes, labels):
                p1 = (int(box[0]*W), int(box[1]*H))
                p2 = (int(box[2]*W), int(box[3]*H))
                cv2.rectangle(origin_image, p1, p2, (0, 255, 0), 1)
                cv2.putText(origin_image, class_inverse_map[label.item()], p1, 1, 1, (0, 255, 0), 1)
        cv2.imshow("img", origin_image)
        cv2.waitKey()


if __name__ == "__main__":
    data_folder_path = r"H:\projectWPD\data"
    dataset    = VOCUtils(data_folder_path).make_dataset(r"VOC2007", r"test.txt", phase="valid")
    model      = SSD(pretrain_path=r"H:\projectWPD\VOC2007_trainval_checkpoint\iteration_120000.pth", n_classes=21)
    detect(dataset, model)