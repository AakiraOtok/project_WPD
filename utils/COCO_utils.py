from utils.lib import *
from utils.augmentations_utils import CustomAugmentation
from pycocotools.coco import COCO

ori_idx        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
handle_idx     = list(range(len(ori_idx)))
name           = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
ori_idx2handle_idx = dict(zip(ori_idx, handle_idx))

COCO_idx2name = dict(zip(handle_idx, name))
COCO_name2idx = dict(zip(name, handle_idx))


def COCO_collate_fn(batches):
    """
    custom dataset với hàm collate_fn để hợp nhất dữ liệu từ batch_size dữ liệu đơn lẻ

    :param batches được trả về từ __getitem__() 
    
    return:
    images, tensor [batch_size, C, H, W]
    bboxes, list [tensor([n_box, 4]), ...]
    labels, list [tensor([n_box]), ...]
    iscrowds, list[tensor[n_box], ...]
    """
    images       = []
    bboxes       = []
    labels       = []
    iscrowds = []

    for batch in batches:
        images.append(batch[0])
        bboxes.append(batch[1])
        labels.append(batch[2])
        iscrowds.append(batch[3])
    
    images = torch.stack(images, dim=0)
    return images, bboxes, labels, iscrowds

class COCO_dataset(data.Dataset):

    def __init__(self, data_folder_path, ann_file, transform, phase="train"):
        self.data_folder_path = data_folder_path
        self.coco = COCO(annotation_file=ann_file)

        # COCO có chứa ảnh không có object, SSD sử dụng hàm loss với tỉ lệ pos:neg,
        # không có object thì đạo hàm = 0, do đó, có thể bỏ qua những ảnh này
        raw_list       = list(self.coco.imgs.keys())
        processed_list = []

        for id in raw_list:
            box = []
            ann_ids = self.coco.getAnnIds(id)
            anns    = self.coco.loadAnns(ann_ids)
            for ann in anns:
                box.append(ann["bbox"])
            if len(box) > 0:
                processed_list.append(id)

        self.img_ids = processed_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx, get_origin_image=False):
        img_id       = self.img_ids[idx]
        img_path     = os.path.join(self.data_folder_path, self.coco.loadImgs(img_id)[0]["file_name"])
        img          = cv2.imread(img_path)
        origin_image = img.copy()

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        bboxes   = []
        labels   = []
        iscrowds = []
        for ann in anns:
            bboxes.append(ann["bbox"])
            labels.append(ori_idx2handle_idx[ann["category_id"]])
            iscrowds.append(ann["iscrowd"])

        bboxes   = np.array(bboxes, dtype=np.float64)
        labels   = np.array(labels)
        iscrowds = np.array(iscrowds)
        bboxes[:, 2:] += bboxes[:, :2]

        img, bboxes, labels, iscrowds = self.transform(img, bboxes, labels, iscrowds, self.phase)

        img          = torch.FloatTensor(img[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous()
        bboxes       = torch.FloatTensor(bboxes)
        labels       = torch.LongTensor(labels)
        iscrowds     = torch.LongTensor(iscrowds)

        if not get_origin_image:   
            return img, bboxes, labels, iscrowds
        else:
            return origin_image, img, bboxes, labels, iscrowds 
        
class COCOUtils():

    def __init__(self, data_folder_path, ann_file):
        self.data_folder_path = data_folder_path
        self.ann_file = ann_file

    def make_dataset(self, transform=CustomAugmentation(), phase="train"):
        dataset = COCO_dataset(self.data_folder_path, self.ann_file, transform, phase)
        return dataset

    def make_dataloader(self, batch_size, shuffle, transform=CustomAugmentation(), collate_fn=COCO_collate_fn, phase='train',num_worker=0, pin_memory=False):
        dataset    = self.make_dataset(transform=transform, phase=phase)
        dataloader = data.DataLoader(dataset, batch_size, shuffle, collate_fn=collate_fn, num_workers=num_worker, pin_memory=pin_memory)
        return dataloader


if __name__ == "__main__":
    train_folder_path  = r"H:\data\COCO\train2014"
    val35k_folder_path = r"H:\data\COCO\val2014"
    train_file         = r"H:\data\COCO\instances_train2014.json"
    val35k_file        = r"H:\data\COCO\instances_valminusminival2014.json"
 
    train  = COCOUtils(train_folder_path, train_file).make_dataset(phase="nothing")
    val35k = COCOUtils(val35k_folder_path, val35k_file).make_dataset(phase="nothing")
    dataset = data.ConcatDataset([train, val35k])
    dataset = train

    for img, bboxes, labels, iscrowds in dataset:
        #img, bboxes, labels, iscrowds = dataset.__getitem__(idx)
        origin_img = img.permute(1, 2, 0)[:, :, (2, 1, 0)].contiguous().numpy()/255
        #print(labels)
        #continue

        H, W, C = origin_img.shape
        for idx, box in enumerate(bboxes):
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            label = COCO_idx2name[labels[idx].item()]
            cv2.rectangle(origin_img, p1, p2, (0, 255, 0), 1)
            cv2.putText(origin_img, label, p1, 1, 1, (0, 255, 0), 1)
        
        cv2.imshow("img", origin_img)
        cv2.waitKey()