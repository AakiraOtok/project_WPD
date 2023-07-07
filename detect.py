from utils.lib import *
from model.SSD300 import SSD
from utils.VOC_utils import VOCUtils, collate_fn, class_inverse_map, class_direct_map
from utils.COCO_utils import COCOUtils, idx2name
from utils.box_utils import Non_Maximum_Suppression, draw_bounding_box

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
        pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset[0], conf[0], conf_threshold=0.2, iou_threshold=0.45, top_k=200)

        draw_bounding_box(origin_image, pred_bboxes, pred_labels, pred_confs, idx2name)
        cv2.imshow("img", origin_image)
        cv2.waitKey()


if __name__ == "__main__":
    data_folder_path = r"H:\projectWPD\data"
    #dataset    = VOCUtils(data_folder_path).make_dataset(r"VOC2007", r"test.txt", phase="valid")

    val35k_folder_path = r"H:\data\COCO\val2014"
    val35k_file        = r"H:\data\COCO\instances_valminusminival2014.json"
    dataset = COCOUtils(val35k_folder_path, val35k_file).make_dataset(phase="valid")

    model      = SSD(pretrain_path=r"H:\projectWPD\COCO_trainval35_checkpoint\iteration_270000.pth", data_train_on="COCO",n_classes=81)
    detect(dataset, model)