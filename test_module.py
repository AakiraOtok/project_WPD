from lib import *
from VOC_utils import *
from box_utils import *

# Test VOCUtils
obj = VOCUtils(r"H:\projectWPD\data")
img_path, ann_path = obj.make_data_path_list(r"VOC2007", r"test.txt")
for img, ann in zip(img_path, ann_path):
    #image = cv2.imread(img)
    boxes, labels, difficults  = obj.read_ann(ann)
    #draw_box(image, boxes, labels)
    #cv2.imshow('img', image)
    #if cv2.waitKey() == ord('q'):
        #break
    print(labels)
    time.sleep(5)

