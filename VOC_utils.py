from lib import *

class VOCUtils():
    """
    Thực hiện việc tạo dataset và dataloader cho VOC dataset
    """

    def __init__(self, data_folder_path):
        """
        data_folder_path : nơi tải VOCdevkit
        """
        self.data_folder_path = data_folder_path

    def make_data_path_list(self, version, file_txt):
        """
        Tạo list image path và annotation path

        args:
        version  : phiên bản của VOC (VOC2007, VOC2012 .etc)
        file_txt : file id của ảnh (trainval.txt, test.txt .etc)

        return:
        Hai list image_path, ann_path
        """
        
        img_template = os.path.join(self.data_folder_path, r"VOCdevkit", version, r"JPEGImages",  r"{}.jpg")
        ann_template = os.path.join(self.data_folder_path, r"VOCdevkit", version, r"Annotations", r"{}.xml")
        
        file_txt_path = os.path.join(self.data_folder_path, r"VOCdevkit", version, r"ImageSets", r"Main", file_txt)

        img_list = []
        ann_list = []

        for line in open(file_txt_path):
            id = line.strip()
            img_list.append(img_template.format(id))
            ann_list.append(ann_template.format(id))
            

        return img_list, ann_list
    
    def read_ann(self, ann_path):
        """
        Đọc thông tin trong annotation

        args:
        ann_path : path của file xml cần đọc

        return:
        bboxes     : np [[xmin, ymin, xmax, ymax], ...]
        labels     : np ['dog', 'cat', ...]
        difficults : np [0, 1, 1, 0 ...]
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
                box.append(int(bnd.find(coor).text) - 1)
            bboxes.append(box)

        return np.array(bboxes), np.array(labels), np.array(difficulties)

