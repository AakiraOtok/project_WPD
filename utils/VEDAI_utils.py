from utils.lib import *
from utils.augmentations_utils import CustomAugmentation

ori_idx    = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 23, 31]
handle_idx = [0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11]
ori_idx2handle_idx = dict(zip([key for key in ori_idx], [value for value in handle_idx]))
hanlde_idx2ori_idx = dict(zip([key for key in handle_idx], [value for value in ori_idx]))
#name       = ['background', 'car', 'truck', 'tractor', 'camping', 'boat', 'motorcycle', 'bus', 'van']
#VEDAI_idx2name = dict(zip([key for key in VEDAI_name2idx.values()], [value for value in VEDAI_name2idx.values()]))

file_list = ["fold01.txt", "fold02.txt", "fold03.txt", "fold04.txt", "fold05.txt", "fold06.txt", "fold07.txt", "fold08.txt", "fold09.txt", "fold10.txt"]
file_test_list = ["fold01test.txt", "fold02test.txt", "fold03test.txt", "fold04test.txt", "fold05test.txt", "fold06test.txt", "fold07test.txt", "fold08test.txt", "fold09test.txt", "fold10test.txt"]
data_folder = r"E:\data"

#for file_name in file_list:
    #path = os.path.join(data_folder, r"Annotations512", file_name)
    #with open(path) as f:
        #for line in f:
            #print(line.strip())
            #time.sleep(1)


# test thử tọa độ, đánh (x1, x2, x3, x4, y1, y2, y3, y4) đánh theo chiều kim đồng hồ bắt đầu từ góc trái trên
def test_coordinate():
    def process(file_name):
        lt = []
        path = os.path.join(data_folder, r"Annotations512", file_name + ".txt")
        with open(path) as f:
            for line in f:
                info = line.strip().split()
                lt += [info]

        return lt


    file_test = r"E:\data\Annotations512\fold01.txt"
    with open(file_test) as f:
        for line in f:
            image = cv2.imread(os.path.join(data_folder, r"Vehicules512", line.strip() + r"_co.png"))
            boxes = process(line.strip())
            for box in boxes:
                p1 = (min(int(box[-5]), int(box[-6]), int(box[-7]), int(box[-8])), min(int(box[-1]), int(box[-2]), int(box[-3]), int(box[-4])))
                p2 = (max(int(box[-5]), int(box[-6]), int(box[-7]), int(box[-8])), max(int(box[-1]), int(box[-2]), int(box[-3]), int(box[-4])))
                cv2.rectangle(image, p1, p2, (0, 0, 255))
            cv2.imshow('img', image)
            cv2.waitKey()
################################################

def VEDAI_collate_fn(batches):
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

def parse_ann(ann_path):
    bboxes = []
    labels = []
    
    with open(ann_path) as file:
        for line in file:
            info  = line.strip().split()
            xmin  = min(float(info[-5]), float(info[-6]), float(info[-7]), float(info[-8]))
            ymin  = min(float(info[-1]), float(info[-2]), float(info[-3]), float(info[-4]))
            xmax  = max(float(info[-5]), float(info[-6]), float(info[-7]), float(info[-8]))
            ymax  = max(float(info[-1]), float(info[-2]), float(info[-3]), float(info[-4]))
            box   = [xmin, ymin, xmax, ymax]
            label = ori_idx2handle_idx[int(info[3])]

            bboxes += [box]
            labels += [label]
        
    difficulties = [0 for i in range(len(labels))]

    return bboxes, labels, difficulties

class VEDAI_dataset(data.Dataset):

    def __init__(self, img_path_list, ann_path_list, transform, phase):
        super().__init__()
        self.img_path_list = img_path_list
        self.ann_path_list = ann_path_list
        self.transform     = transform
        self.phase         = phase

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index, get_origin_image=False):
        image                        = cv2.imread(self.img_path_list[index])
        origin_image                 = image.copy()
        bboxes, labels, difficulties = parse_ann(self.ann_path_list[index])

        bboxes       = np.array(bboxes)
        labels       = np.array(labels)
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
        

class VEDAI_Utils():

    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    def make_data_path_list(self, fold_file):
        
        img_template = os.path.join(self.data_folder_path, r"Vehicules512", r"{}_co.png")
        ann_tmeplate = os.path.join(self.data_folder_path, r"Annotations512", r"{}.txt")

        img_list = []
        ann_list = []

        fold_file = os.path.join(self.data_folder_path, r"Annotations512", fold_file)

        with open(fold_file) as file:
            for line in file:
                line = line.strip()
                img_list.append(img_template.format(line))
                ann_list.append(ann_tmeplate.format(line))
        
        return img_list, ann_list
        
    def make_dataset(self, fold_file, transform=CustomAugmentation(), phase='train'):
        img_path_list, ann_path_list = self.make_data_path_list(fold_file)
        dataset = VEDAI_dataset(img_path_list, ann_path_list, transform, phase)
        return dataset
    
    def make_dataloader(self, fold_file, batch_size, shuffle, transform=CustomAugmentation(), collate_fn=VEDAI_collate_fn, phase='train',num_worker=0, pin_memory=False):
        dataset    = self.make_dataset(fold_file, transform, phase)
        dataloader = data.DataLoader(dataset, batch_size, shuffle, num_workers=num_worker, collate_fn=collate_fn, pin_memory=pin_memory) 
        return dataloader
    
if __name__ == "__main__":
    data_folder_path = r"E:\data"
    vedai = VEDAI_Utils(data_folder_path)
    dataloader = vedai.make_dataloader(r"fold01.txt", 1, 0, phase="valid")
    for images, bboxes, labels, difficulties in dataloader:
        img = images.squeeze(0).permute(1, 2, 0).contiguous()[:, :, (2, 1, 0)].cpu().numpy()
        bboxes = bboxes[0]
        labels = labels[0]
        difficulties = difficulties[0]
        
        H, W, C = img.shape
        H -= 1
        W -= 1

        for box, label, difficult in zip(bboxes, labels, difficulties):
            p1 = (int(box[0]*W), int(box[1]*H))
            p2 = (int(box[2]*W), int(box[3]*H))
            cv2.rectangle(img, p1, p2, (0, 0, 255), 1)
        cv2.imshow("img", img)
        cv2.waitKey()