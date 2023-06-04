from lib import *

def draw_box(image, bboxes, labels, confidence_score=None, color=(0, 255, 0, 0.1), thickness=1, line_type=cv2.LINE_AA):
    """
    Vẽ bouding box

    agrs:
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

