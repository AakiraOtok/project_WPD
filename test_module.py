from lib import *
from VOC_utils import VOCUtils, class_inverse_map
data_folder_path = r"H:\projectWPD\data"
voc = VOCUtils(data_folder_path)

dataset = voc.make_dataset(version=r"VOC2007", file_txt=r"test.txt", phase='test')

for i in range(dataset.__len__()):
    img, bboxes, labels, difficulties = dataset.__getitem__(i)
    img = img.permute(1, 2, 0).contiguous()[:, :, (2, 1, 0)].contiguous().numpy()

    for box, label, difficult in zip(bboxes, labels, difficulties):
        p1 = (int(box[0]*300), int(box[1]*300))
        p2 = (int(box[2]*300), int(box[3]*300))

        cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
        text = class_inverse_map[label.item()] + " " + str(difficult.item())
        cv2.putText(img, text, p1, 1, 1, (0, 255, 0), 1)

    cv2.imshow('img', img)
    cv2.waitKey()