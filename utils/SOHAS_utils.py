from utils.lib import *
from utils.augmentations_utils import CustomAugmentation

SOHAS_idx2name = {
    1 : "pistol",
    2 : "smartphone",
    3 : "knife",
    4 : "monedero",
    5 : "billete",
    6 : "tarjeta"
}

SOHAS_name2idx = {
    "pistol"     : 1,
    "smartphone" : 2,
    "knife"      : 3,
    "monedero"   : 4,
    "billete"    : 5,
    "tarjeta"    : 6
}

def collate_fn(batches):
    """
    custom dataset với hàm collate_fn để hợp nhất dữ liệu từ batch_size dữ liệu đơn lẻ

    :param batches được trả về từ __getitem__() 
    
    return:
    images, tensor [batch_size, C, H, W]
    bboxes, list [tensor([n_box, 4]), ...]
    labels, list [tensor([n_box]), ...]
    difficulties, list[tensor[n_box], ...]
    """
    images       = []
    bboxes       = []
    labels       = []
    difficulties = []

    for batch in batches:
        images.append(batch[0])
        bboxes.append(batch[1])
        labels.append(batch[2])
        difficulties.append(batch[3])
    
    images = torch.stack(images, dim=0)
    return images, bboxes, labels, difficulties

def read_ann(ann_path):
    """
    Đọc thông tin trong annotation

    args:
    ann_path : path của file xml cần đọc

    return:
    bboxes     : list [[xmin, ymin, xmax, ymax], ...]
    labels     : list ['dog', 'cat', ...]
    difficults : list [0, 1, 1, 0 ...]
    """

    tree = ET.parse(ann_path)
    root = tree.getroot()

    coors = ['xmin', 'ymin', 'xmax', 'ymax']

    bboxes        = []
    labels        = []
    difficulties  = []

    for obj in root.iter('object'):
        # Tên của obj trong box
        name = obj.find('name').text.lower().strip()
        labels.append(name)

        # Độ khó 
        difficult = int(obj.find('difficult').text)
        difficulties.append(difficult)

        # Toạ độ
        bnd = obj.find("bndbox")
        box = []
        for coor in coors:
            box.append(float(bnd.find(coor).text) - 1)
        bboxes.append(box)

    return bboxes, labels, difficulties

class SOHAS_dataset(data.Dataset):

    def __init__(self, data_folder_path, folder, transform, phase):
        super().__init__()

        self.transform = transform
        self.phase     = phase
        
        self.img_template = os.path.join(data_folder_path, r"SOHAS", folder, r"images", r"{}.jpg")
        self.ann_template = os.path.join(data_folder_path, r"SOHAS", folder, r"annotations",r"{}.xml")
        self.id_list = []

        id_path = os.path.join(data_folder_path, r"SOHAS", folder, r"images")
        for idx in os.listdir(id_path):
            self.id_list.append(idx.strip().split('.')[0])

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx, get_origin_image=False):
        img_path = self.img_template.format(self.id_list[idx])
        ann_path = self.ann_template.format(self.id_list[idx])

        image        = cv2.imread(img_path)
        
        origin_image = image.copy()

        bboxes, labels, difficulties = read_ann(ann_path)
        temp = []
        for label in labels:
            temp.append(SOHAS_name2idx[label])
        bboxes       = np.array(bboxes)
        labels       = np.array(temp)
        difficulties = np.array(difficulties)

        image, bboxes, labels, difficulties = self.transform(image, bboxes, labels, difficulties, self.phase) 

        image        = torch.FloatTensor(image[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous()
        bboxes       = torch.FloatTensor(bboxes)
        labels       = torch.LongTensor(labels)
        difficulties = torch.LongTensor(difficulties)

        if not get_origin_image:   
            return image, bboxes, labels, difficulties
        else:
            return origin_image, image, bboxes, labels, difficulties
        
class SOHASUtils():

    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    def make_dataset(self, folder, transform=CustomAugmentation(), phase='train'):
        dataset = SOHAS_dataset(data_folder_path=self.data_folder_path, folder=folder, transform=transform, phase=phase)
        return dataset

    def make_dataloader(self, folder, batch_size, shuffle, transform=CustomAugmentation(), collate_fn=collate_fn, phase='train',num_worker=0, pin_memory=False):
        dataset    = self.make_dataset(folder, transform=transform, phase=phase)
        dataloader = data.DataLoader(dataset, batch_size, shuffle, num_workers=num_worker, collate_fn=collate_fn, pin_memory=pin_memory)
        return dataloader


if __name__ == "__main__":
    data_folder_path = r"H:\data"
    T = SOHAS_dataset(data_folder_path, r'train', CustomAugmentation(size=512), phase='valid')

    for idx in range(T.__len__()):
        print(idx)
        orin_image, image, bboxes, labels, difficulties = T.__getitem__(idx, get_origin_image=True)
        
        H, W, C = orin_image.shape

        for i, box in enumerate(bboxes):
            #print(box)
            box[0] *= W
            box[1] *= H
            box[2] *= W
            box[3] *= H

            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            #print(p1,p2)
            #print("//////////////////")

            cv2.rectangle(orin_image, p1, p2, (0, 255, 0), 1, 1)
            cv2.putText(orin_image, SOHAS_idx2name[labels[i].item()], p1, 1, 1, (0, 255, 0), 1, 1)

        cv2.imshow('img', orin_image)
        cv2.waitKey()
