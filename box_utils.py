from lib import *

def draw_box(image, bboxes, labels, confidence_score=None, color=(0, 255, 0, 0.1), thickness=1, line_type=cv2.LINE_AA):
    """
    Vẽ bouding box

    param:
    image  : numpy array [H, W, C]
    bboxes : numpy arra
    labels :
    confidence_score :
    """

    for box, label in zip(bboxes, labels):
        p1 = (box[0], box[1])
        p2 = (box[2], box[3])

        cv2.rectangle(image, p1, p2, color, thickness, line_type)
        cv2.putText(image, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def pascalVOC_style(boxes):
    """
    Chuyển từ yolo_style sang pascalVOC_style : [cx, cy, w, h] -> [xmin, ymin, xmax, ymax]
     
    :param boxes, tensor [nbox, 4]
    
    return:
    boxes tensor [nbox, 4]
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), dim=1)

def yolo_style(boxes):
    """
    Chuyển từ pascalVOC_style sang yolo_style : [xmin, ymin, xmax, ymax] -> [cx, cy, w, h]
      
    :param boxes, tensor [nbox, 4]

    return:
    boxes tensor [nbox, 4]
    """
    return torch.cat(((boxes[:, :2] + boxes[:, 2:])/2, boxes[:, 2:] - boxes[:, :2]), dim=1)

# encode (giải thích trên hackmd)
def encode_variance(dboxes, bboxes, variances=[0.1, 0.2]):
    return torch.cat(((bboxes[:, :2] - dboxes[:, :2])/(dboxes[:, 2:]*variances[0]), 
                      torch.log(bboxes[:, 2:]/dboxes[:, 2:])/variances[1]), dim=1)

# decode (giải thích trên hackmd)
def decode_variance(dboxes, loc, variances=[0.1, 0.2]):
    return torch.cat((loc[:, :2]*dboxes[:, 2:]*variances[0] + dboxes[:, :2],
                     torch.exp(loc[:, 2:]*variances[1])*dboxes[:, 2:]), dim=1)

def intersect(box_a, box_b):
    """
    Trả về diện tích giao nhau của các box
    các box đang ở dạng [xmin, ymin, xmax, ymax] chuẩn hóa [0...1]
    
    :param box_a, tensor [nbox_a, 4]
    :param box_b, tensor [nbox_b, 4]
    
    return:
    inter tensor, [nbox_a, nbox_b]
    """
    A = box_a.size(0)
    B = box_b.size(0)

    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, 0)

    return inter[:, :, 0] *inter[:, :, 1]

def jaccard(box_a, box_b):
    """
     Trả về hệ số jaccard của các box
     các box đang ở dạng [xmin, ymin, xmax, ymax] chuẩn hóa [0...1]
      
    :param box_a, tensor [nbox_a, 4]
    :param box_b, tensor [nbox_b, 4]
    
    return:
    overlap [nbox_a, nbox_b]
    """
    A = box_a.size(0)
    B = box_b.size(0)

    temp = box_a[:, 2:] - box_a[:, :2]
    area_box_a = (temp[:, 0]*temp[:, 1]).unsqueeze(1).expand(A, B)
    temp = box_b[:, 2:] - box_b[:, :2]
    area_box_b = (temp[:, 0]*temp[:, 1]).unsqueeze(0).expand(A, B)

    area = area_box_a + area_box_b
    inter = intersect(box_a, box_b)

    return inter/(area - inter)