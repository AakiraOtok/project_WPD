from lib import *
from VOC_utils import *
from box_utils import *
#from augmentations_utils import *
from utils import *

# Test VOCUtils
#obj = VOCUtils(r"H:\projectWPD\data")
#img_path, ann_path = obj.make_data_path_list(r"VOC2007", r"test.txt")
#for img, ann in zip(img_path, ann_path):
    #image = cv2.imread(img)
    #boxes, labels, difficults  = obj.read_ann(ann)
    #draw_box(image, boxes, labels)
    #cv2.imshow('img', image)
    #if cv2.waitKey() == ord('q'):
        #break
    #image = Image.open(img)
    #image.show()
    #time.sleep(5)
img = Image.open(r"C:\Users\eguit\Pictures\Saved Picture\cute-1-300x300.png", 'r')
img = img.convert('RGB')
img = np.asarray(img)
print(img[0, 0, 0])

while cv2.waitKey() != ord('q'):
    img = cv2.imread(r"C:\Users\eguit\Pictures\Saved Picture\cute-1-300x300.png")
    img = torch.tensor(img[:, :, (2, 1, 0)]).permute(2, 0, 1).contiguous()/255
    #img = Image.open(r"C:\Users\eguit\Pictures\Saved Picture\cute-1-300x300.png")
    #img = img.convert('RGB')
    #img = FT.to_tensor(img)
    bboxes = torch.FloatTensor([[0, 0, 299, 299]])
    labels = torch.tensor([1])
    difficulties = torch.tensor([1])

    #img, bboxes = expand(img, bboxes, mean)
    img, bboxes, labels, difficulties = transform(img, bboxes, labels, difficulties, 'TRAIN')
    #img, bboxes = resize(img, bboxes)
    #print(img[0, 0, 0])
    #img = photometric_distort(img)
    #print(img[0, 0, 0])
    img = img.permute(1, 2, 0).contiguous().numpy()[:, :, (2, 1, 0)]
    cv2.imshow('img', img)
    #print(img)
    #time.sleep(5)

