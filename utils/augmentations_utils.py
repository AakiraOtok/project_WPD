import torch
import cv2
import numpy as np
import types
import random
import torchvision.transforms.functional as FT
from PIL import Image


#def intersect(box_a, box_b):
    #max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    #min_xy = np.maximum(box_a[:, :2], box_b[:2])
    #inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    #return inter[:, 0] * inter[:, 1]


#def jaccard_numpy(box_a, box_b):
    #"""Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    #is simply the intersection over union of two boxes.
    #E.g.:
        #A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    #Args:
        #box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        #box_b: Single bounding box, Shape: [4]
    #Return:
        #jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    #"""
    #inter = intersect(box_a, box_b)
    #area_a = ((box_a[:, 2]-box_a[:, 0]) *
              #(box_a[:, 3]-box_a[:, 1]))  # [A,B]
    #area_b = ((box_b[2]-box_b[0]) *
              #(box_b[3]-box_b[1]))  # [A,B]
    #union = area_a + area_b - inter
    #return inter / union  # [A,B]


#class Compose(object):
    #"""Composes several augmentations together.
    #Args:
        #transforms (List[Transform]): list of transforms to compose.
    #Example:
        #>>> augmentations.Compose([
        #>>>     transforms.CenterCrop(10),
        #>>>     transforms.ToTensor(),
        #>>> ])
    #"""

    #def __init__(self, transforms):
        #self.transforms = transforms

    #def __call__(self, img, boxes=None, labels=None, difficulties=None):
        #for t in self.transforms:
            #img, boxes, labels, difficulties = t(img, boxes, labels, difficulties)
        #return img, boxes, labels, difficulties


#class Lambda(object):
    #"""Applies a lambda as a transform."""

    #def __init__(self, lambd):
        #assert isinstance(lambd, types.LambdaType)
        #self.lambd = lambd

    #def __call__(self, img, boxes=None, labels=None, difficulties=None):
        #return self.lambd(img, boxes, labels, difficulties)


#class ConvertFromInts(object):
    #"""
    #chuyển ảnh từ int sang float32
    #"""
    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #return image.astype(np.float32), boxes, labels, difficulties


#class SubtractMeans(object):
    #def __init__(self, mean):
        #self.mean = np.array(mean, dtype=np.float32)

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #image = image.astype(np.float32)
        #image -= self.mean
        #return image.astype(np.float32), boxes, labels, difficulties
    
#class Normalize():
    #def __init__(self, mean, std):
        #self.mean = np.array(mean)
        #self.std  = np.array(std)

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #image = image.astype(np.float32)/255
        #image = (image - self.mean)/self.std
        #return image, boxes, labels, difficulties


#class ToAbsoluteCoords(object):
    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #height, width, channels = image.shape
        #boxes[:, 0] *= width
        #boxes[:, 2] *= width
        #boxes[:, 1] *= height
        #boxes[:, 3] *= height

        #return image, boxes, labels, difficulties


#class ToPercentCoords(object):
    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #if boxes is not None:
            #height, width, channels = image.shape
            #boxes[:, 0] /= width
            #boxes[:, 2] /= width
            #boxes[:, 1] /= height
            #boxes[:, 3] /= height

        #return image, boxes, labels, difficulties


#class Resize(object):
    #def __init__(self, size=300):
        #self.size = size

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #image = cv2.resize(image, (self.size,
                                 #self.size))
        #return image, boxes, labels, difficulties


#class RandomSaturation(object):
    #def __init__(self, lower=0.5, upper=1.5):
        #self.lower = lower
        #self.upper = upper
        #assert self.upper >= self.lower, "contrast upper must be >= lower."
        #assert self.lower >= 0, "contrast lower must be non-negative."

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #if np.random.randint(2):
            #image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        #return image, boxes, labels, difficulties


#class RandomHue(object):
    #def __init__(self, delta=18.0):
        #assert delta >= 0.0 and delta <= 360.0
        #self.delta = delta

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #if np.random.randint(2):
            #image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            #image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            #image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        #return image, boxes, labels, difficulties


#class RandomLightingNoise(object):
    #def __init__(self):
        #self.perms = ((0, 1, 2), (0, 2, 1),
                      #(1, 0, 2), (1, 2, 0),
                      #(2, 0, 1), (2, 1, 0))

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #if np.random.randint(2):
            #swap = self.perms[np.random.randint(len(self.perms))]
            #shuffle = SwapChannels(swap)  # shuffle channels
            #image = shuffle(image)
        #return image, boxes, labels, difficulties


#class ConvertColor(object):
    #def __init__(self, current='BGR', transform='HSV'):
        #self.transform = transform
        #self.current = current

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #if self.current == 'BGR' and self.transform == 'HSV':
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #elif self.current == 'HSV' and self.transform == 'BGR':
            #image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        #else:
            #raise NotImplementedError
        #return image, boxes, labels, difficulties


#class RandomContrast(object):
    #def __init__(self, lower=0.5, upper=1.5):
        #self.lower = lower
        #self.upper = upper
        #assert self.upper >= self.lower, "contrast upper must be >= lower."
        #assert self.lower >= 0, "contrast lower must be non-negative."

    ## expects float image
    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #if np.random.randint(2):
            #alpha = np.random.uniform(self.lower, self.upper)
            #image *= alpha
        #return image, boxes, labels, difficulties


#class RandomBrightness(object):
    #def __init__(self, delta=32):
        #assert delta >= 0.0
        #assert delta <= 255.0
        #self.delta = delta

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #if np.random.randint(2):
            #delta = np.random.uniform(-self.delta, self.delta)
            #image += delta
        #return image, boxes, labels, difficulties


#class ToCV2Image(object):
    #def __call__(self, tensor, boxes=None, labels=None, difficulties=None):
        #return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels, difficulties


#class ToTensor(object):
    #def __call__(self, cvimage, boxes=None, labels=None, difficulties=None):
        #return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels, difficulties


#class RandomSampleCrop(object):
    #"""Crop
    #Arguments:
        #img (Image): the image being input during training
        #boxes (Tensor): the original bounding boxes in pt form
        #labels (Tensor): the class labels for each bbox
        #mode (float tuple): the min and max jaccard overlaps
    #Return:
        #(img, boxes, classes)
            #img (Image): the cropped image
            #boxes (Tensor): the adjusted bounding boxes in pt form
            #labels (Tensor): the class labels for each bbox
    #"""
    #def __init__(self):
        #self.sample_options = (
            ## using entire original input image
            #None,
            ## sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            #(0.1, None),
            #(0.3, None),
            #(0.7, None),
            #(0.9, None),
            ## randomly sample a patch
            #(None, None),
        #)

    #def __call__(self, image, boxes=None, labels=None, difficulties=None):
        #height, width, _ = image.shape
        #while True:
            ## randomly choose a mode
            #mode = random.choice(self.sample_options)
            #if mode is None:
                #return image, boxes, labels, difficulties

            #min_iou, max_iou = mode
            #if min_iou is None:
                #min_iou = float('-inf')
            #if max_iou is None:
                #max_iou = float('inf')

            ## max trails (50)
            #for _ in range(50):
                #current_image = image

                #w = np.random.uniform(0.3 * width, width)
                #h = np.random.uniform(0.3 * height, height)

                ## aspect ratio constraint b/t .5 & 2
                #if h / w < 0.5 or h / w > 2:
                    #continue

                #left = np.random.uniform(width - w)
                #top = np.random.uniform(height - h)

                ## convert to integer rect x1,y1,x2,y2
                #rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                ## calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                #overlap = jaccard_numpy(boxes, rect)

                ## is min and max overlap constraint satisfied? if not try again
                ## a bug was adressed at https://github.com/amdegroot/ssd.pytorch/issues/119
                #if overlap.max() < min_iou or overlap.min() > max_iou:
                    #continue 

                ## cut the crop from the image
                #current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              #:]

                ## keep overlap with gt box IF center in sampled patch
                #centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                ## mask in all gt boxes that above and to the left of centers
                #m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                ## mask in all gt boxes that under and to the right of centers
                #m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                ## mask in that both m1 and m2 are true
                #mask = m1 * m2

                ## have any valid boxes? try again if not
                #if not mask.any():
                    #continue

                ## take only matching gt boxes
                #current_boxes = boxes[mask, :].copy()

                ## take only matching gt labels
                #current_labels = labels[mask]

                ## add difficulties
                #current_difficulties = difficulties[mask]

                ## should we use the box left and top corner or the crop's
                #current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  #rect[:2])
                ## adjust to crop (by substracting crop's left,top)
                #current_boxes[:, :2] -= rect[:2]

                #current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  #rect[2:])
                ## adjust to crop (by substracting crop's left,top)
                #current_boxes[:, 2:] -= rect[:2]

                #return current_image, current_boxes, current_labels, current_difficulties


#class Expand(object):
    #def __init__(self, mean):
        #self.mean = mean

    #def __call__(self, image, boxes, labels, difficulties):
        #if np.random.randint(2):
            #return image, boxes, labels, difficulties

        #height, width, depth = image.shape
        #ratio = np.random.uniform(1, 4)
        #left = np.random.uniform(0, width*ratio - width)
        #top = np.random.uniform(0, height*ratio - height)

        #expand_image = np.zeros(
            #(int(height*ratio), int(width*ratio), depth),
            #dtype=image.dtype)
        #expand_image[:, :, :] = self.mean
        #expand_image[int(top):int(top + height),
                     #int(left):int(left + width)] = image
        #image = expand_image

        #boxes = boxes.copy()
        #boxes[:, :2] += (int(left), int(top))
        #boxes[:, 2:] += (int(left), int(top))

        #return image, boxes, labels, difficulties


#class RandomMirror(object):
    #def __call__(self, image, boxes, classes, difficulties):
        #_, width, _ = image.shape
        #if np.random.randint(2):
            #image = image[:, ::-1]
            #boxes = boxes.copy()
            #boxes[:, 0::2] = width - boxes[:, 2::-2]
        #return image, boxes, classes, difficulties


#class SwapChannels(object):
    #"""Transforms a tensorized image by swapping the channels in the order
     #specified in the swap tuple.
    #Args:
        #swaps (int triple): final order of channels
            #eg: (2, 1, 0)
    #"""

    #def __init__(self, swaps):
        #self.swaps = swaps

    #def __call__(self, image):
        #"""
        #Args:
            #image (Tensor): image tensor to be transformed
        #Return:
            #a tensor with channels swapped according to swap
        #"""
        ## if torch.is_tensor(image):
        ##     image = image.data.cpu().numpy()
        ## else:
        ##     image = np.array(image)
        #image = image[:, :, self.swaps]
        #return image


#class PhotometricDistort(object):
    #def __init__(self):
        #self.pd = [
            #RandomContrast(),
            #ConvertColor(transform='HSV'),
            #RandomSaturation(),
            #RandomHue(),
            #ConvertColor(current='HSV', transform='BGR'),
            #RandomContrast()
        #]
        #self.rand_brightness = RandomBrightness()
        #self.rand_light_noise = RandomLightingNoise()

    #def __call__(self, image, boxes, labels, difficulties):
        #im = image.copy()
        #im, boxes, labels, difficulties = self.rand_brightness(im, boxes, labels, difficulties)
        #if np.random.randint(2):
            #distort = Compose(self.pd[:-1])
        #else:
            #distort = Compose(self.pd[1:])
        #im, boxes, labels, difficulties = distort(im, boxes, labels, difficulties)
        #return self.rand_light_noise(im, boxes, labels, difficulties)


#class SSDAugmentation(object):
    #def __init__(self, size=300, mean=(104, 117, 123)):
        #self.mean = mean
        #self.size = size
        #self.augment = Compose([
            #ConvertFromInts(),
            #ToAbsoluteCoords(),
            #PhotometricDistort(),
            #Expand(self.mean),
            #RandomSampleCrop(),
            #RandomMirror(),
            #ToPercentCoords(),
            #Resize(self.size),
            #SubtractMeans(self.mean)
        #])

    #def __call__(self, img, boxes, labels, difficulties):
        #return self.augment(img, boxes, labels, difficulties)

#class CustomAugmentation():
    ##def __init__(self, phase="train", size=300, mean=[123./255, 117./255, 104./255], std=[1., 1., 1.]):
    #def __init__(self, phase="train", size=300, mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
        #assert(phase in ("train", "valid"))
        #self.mean = mean
        #self.size = size
        #self.train_augment = Compose([
                #ConvertFromInts(),
                #PhotometricDistort(),
                #Expand(self.mean),
                #RandomSampleCrop(),
                #RandomMirror(),
                #ToPercentCoords(),
                #Resize(self.size),
                ##SubtractMeans(self.mean)
                #Normalize(mean, std)
            #])
        #self.valid_augment = Compose([
                #ConvertFromInts(),
                #ToPercentCoords(),
                #Resize(self.size),
                ##SubtractMeans(self.mean)
                #Normalize(mean, std)
            #])

    #def __call__(self, img, boxes=None, labels=None, difficulties=None, phase="train"):
        #if phase == "train":
            #return self.train_augment(img, boxes, labels, difficulties)
        #elif phase == "valid":
            #return self.valid_augment(img, boxes, labels, difficulties)
        #elif phase == "nothing":
            #return img, boxes, labels, difficulties

def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image

from utils.lib import *
class CustomAugmentation():
    def __init__(self, phase="train", size=300, mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
        self.size=size
        pass

    def __call__(self, image, boxes, labels, difficulties, phase):
        """
        Apply the transformations above.

        :param image: image, a PIL Image
        :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
        :param labels: labels of objects, a tensor of dimensions (n_objects)
        :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
        :param phase: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
        :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
        """
        #########################################################################

        # Convert cv2 image to numpy array
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert numpy array to PIL image
        image = Image.fromarray(image)

        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        difficulties = torch.tensor(difficulties)

        #########################################################################


        assert phase in {'train', 'valid'}

        # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
        # see: https://pytorch.org/docs/stable/torchvision/models.html
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        new_image = image
        new_boxes = boxes
        new_labels = labels
        new_difficulties = difficulties
        # Skip the following operations for evaluation/testing
        if phase == 'train':
            # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
            new_image = photometric_distort(new_image)

            # Convert PIL image to Torch tensor
            new_image = FT.to_tensor(new_image)

            # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
            # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
            if random.random() < 0.5:
                new_image, new_boxes = expand(new_image, boxes, filler=mean)

            # Randomly crop image (zoom in)
            new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                            new_difficulties)

            # Convert Torch tensor to PIL image
            new_image = FT.to_pil_image(new_image)

            # Flip image with a 50% chance
            if random.random() < 0.5:
                new_image, new_boxes = flip(new_image, new_boxes)

        # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
        new_image, new_boxes = resize(new_image, new_boxes, dims=(self.size, self.size))

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
        new_image = FT.normalize(new_image, mean=mean, std=std)

        #####################################################################################

        # Convert tensor to numpy array
        new_image = np.array(new_image.permute(1, 2, 0).contiguous())

        # Convert numpy array to cv2 image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

        new_boxes = new_boxes.numpy()
        new_labels = new_labels.numpy()
        new_difficulties = new_difficulties.numpy()

        #####################################################################################

        return new_image, new_boxes, new_labels, new_difficulties