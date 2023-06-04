from lib import *
from box_utils import jaccard

# Các hàm được lấy từ : https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/tree/master

def photometric_distort(image):
    """
    Làm biến dạng (distort) độ sáng (brightness), độ tương phản (contrast), độ bão hòa (saturation) và màu sắc (hue)
    với xác suất 50%, thứ tự thực hiện các thao tác là ngẫu nhiên

    param:
    image : tensor [C, H, W] (RGB)

    return:
    new_image : tensor [C, H, W] (RGB)

    """
    
    new_image = image.clone()

    distortions = [
        FT.adjust_brightness,
        FT.adjust_contrast,
        FT.adjust_saturation,
        FT.adjust_hue
    ]

    random.shuffle(distortions)

    for manip in distortions:
        if random.random() < 0.5 : 
            if manip.__name__ == 'adjust_hue':
                # Caffe repo sử dụng hue delta = 18, ta chia cho 255 bởi vì Pytorch cần giá trị chuẩn hóa
                adjust_factor = random.uniform(-18/255., 18/255)

            else:
                #Caffe repo sử dụng 'lower' và 'upper' là 0.5 và 1.5 cho brightness, contrast và saturation
                adjust_factor = random.uniform(0.5, 1.5)

            new_image = manip(new_image, adjust_factor)

    return new_image

def expand(image, bboxes, filler):
    """
    Thực hiện thu nhỏ hình ảnh (zoom out) bằng cách đặt ảnh gốc vào một khung lớn hơn, khoảng trống được lấp lại bằng filler

    Giúp học được cách phát hiện các vật thể nhỏ tốt hơn

    param:
    image  : tensor [C, H, W]
    bboxes : tensor [n_objects, 4] [xmin, ymin, xmax, ymax]
    filler : dùng để lấp khoảng trống, list [R, G, B]

    return:
    image  : tensor [C, H, W]
    bboxes : tensor [n_objects, 4] [xmin, ymin, xmax, ymax]
    """

    C, original_h, original_w = image.shape 

    min_scale = 1
    max_scale = 4
    scale = random.uniform(min_scale, max_scale)

    new_h = int(original_h*scale)
    new_w = int(original_w*scale)

    filler =  torch.FloatTensor(filler) # [3]
    new_image = torch.ones((C, new_h, new_w)) * filler.unsqueeze(1).unsqueeze(1) # [3, new_h, new_w]
    # Không dùng expand() như new_image = filler.unsqueeze(1).unsquezee(1).expand(3, new_h, new_w)
    # vì tất cả giá trị expand đều dùng chung bộ nhớ, đổi một pixel sẽ đổi tất cả


    left  = random.randint(0, new_w - original_w)
    right = left + original_w

    top    = random.randint(0, new_h - original_h)
    bottom = top + original_h

    # Đặt ảnh gốc vào bên trong ảnh mới
    new_image[:, top:bottom, left:right] = image

    # Tính lại vị trí của bboxes
    new_bboxes = bboxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_bboxes

def random_crop(image, bboxes, labels, difficulties):
    """
    Thực hiện cắt ngẫu nhiên (random crop) như đã nếu trong paper gốc. Giúp cho việc học xác định các đối tượng lớn và bị che
    chỉ xuất hiện một phần (partial objects).

    Lưu ý rằng có thể có vài objects sẽ được loại bỏ hoàn toàn

    :param image, tensor [C, H, W]
    :param bboxes, tensor [n_objects, 4]
    :param labels, tensor [n_objects]
    :param difficulties, tensor [n_objects]

    return: 
    new_image, tensor [C, new_h, new_w]
    new_bboxes, tensor [new_n_objects, 4]
    new_labels, tensor [new_n_objects]
    new_difficulties, tensor [new_n_objects]
    """

    C, original_h, original_w = image.shape

    while True:

        # Các mức min overlap ngẫu nhiên, tức là yêu cầu ít nhất có một box > min overlap với croped image
        min_overlap = random.choice([0., 0.1, 0.3, 0.5, 0.7, 0.9, None])

        if min_overlap is None:
            return image, bboxes, labels, difficulties
        
        # Thử 50 lần cho min_overlap này
        # Điều này không được đề cập trong paper nhưng tác giả đã lựa chọn max_trials=50 trong repo chính thức của mình
        max_trials = 50
        for _ in range(max_trials):
            # Scale phải trong khoảng [0.3, 1]
            # paper đề cập nó là [0.1, 1], nhưng thực sự trong repo chính thức con số này là [0.3, 1]
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)

            new_h = int(original_h*scale_h)
            new_w = int(original_w*scale_w)

            # Tỉ lệ khung hình phải trong đoạn [0.5, 2]
            aspect_ratio = new_h/new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            left   = random.randint(0, original_w - new_w)
            right  = left + new_w
            top    = random.randint(0, original_h - new_h)
            bottom = top + new_h

            crop   = torch.FloatTensor([left, top, right, bottom])

            overlap    = jaccard(crop.unsqueeze(0), bboxes).squeeze(0) # [nbox]
            overlap    = overlap.max().item() #scalar

            if overlap < min_overlap:
                continue

            new_image = image[:, top:bottom, left:right]

            # center của các box
            center_bboxes = (bboxes[:, 2:] - bboxes[:, :2])/2.
            # lọc các box có center ở trong vùng crop
            mask = (center_bboxes[:, 0] > left)*(center_bboxes[:, 0] < right)*(center_bboxes[:, 1] > top)*(center_bboxes[:, 1] < bottom)

            if not mask.any():
                continue

            new_bboxes       = bboxes[mask]
            new_labels       = labels[mask]
            new_difficulties = difficulties[mask]

            new_bboxes[:, :2]  = torch.max(new_bboxes[:, :2], crop[:2])
            new_bboxes[:, :2] -= crop[:2]
            new_bboxes[:, 2:]  = torch.min(new_bboxes[:, 2:], crop[2:])
            new_bboxes[:, 2:] -= crop[:2]

            return new_image, new_bboxes, new_labels, new_difficulties 

def flip(image, bboxes):
    """
    Lật ảnh lại theo chiều ngang

    :param image, tensor [C, H, W]
    :param bboxes, tensor [n_objects, 4]

    return:
    new_image, tensor [C, H, W]
    new_bboxes, tensor [n_objects, 4]
    """

    C, H, W = image.shape
    new_image = FT.hflip(image)

    new_bboxes = bboxes
    new_bboxes[:, 0] = W - new_bboxes[:, 0] - 1  # -1 bởi vì kích thước ảnh là H thì trục tọa độ là [0 ... H - 1]
    new_bboxes[:, 2] = W - new_bboxes[:, 2] - 1

    # xmin và xmax sau phép biến đổi đã đổi chỗ cho nhau, cần đổi lại
    new_bboxes = new_bboxes[:, [2, 1, 0, 3]]

    return new_image, new_bboxes

def resize(image, bboxes, dims=(300, 300), return_percent_coords=True):
    """
    Thay đổi kích thước của bức ảnh. SSD sử dụng size=(300, 300) 

    :param image, tensor [C, H, W]
    :param bboxes, tensor [n_objects, 4]

    return:
    new_image, tensor [C, H, W]
    new_bboxes, tensor [n_objects, 4] chuẩn hóa [0, 1]
    """

    C, original_h, original_w = image.shape
    new_image = FT.resize(image, dims)

    # chia cho [W, H, W, H], không phải [W - 1, H - 1, W - 1, H - 1] vì khi khôi phục lại thì ta nhân W, H
    old_dims   = torch.FloatTensor([original_w, original_h, original_w, original_h]).unsqueeze(0)
    new_bboxes = bboxes/old_dims

    if return_percent_coords:
        new_dims   = torch.FloatTensor([dims[0], dims[1], dims[0], dims[1]]).unsqueeze(0) 
        new_bboxes = new_bboxes*new_dims 

    return new_image, new_bboxes


def transform(image, bboxes, labels, difficulties, phase='train'):
    """
    Áp dụng các bước transform như trong paper:

    - Ngẫu nhiên điều chỉnh brightness, contrast, saturation và hue với tỉ lệ 50% cho mỗi thao tác và thứ tự ngẫu nhiên
    - Ngẫu nhiên phóng to ảnh với tỉ lệ [1, 4]
    - Random crop ảnh, tỉ lệ khung hình phải trong khoảng [0...2], các mức min_overlap=[0., 0.1, 0.3, 0.5, 0.7, 0.9, None]
    - Ngẫu nhiên flip dọc với tỉ lệ 50%
    - Resize ảnh lại về (300, 300)
    - Chuyển tất cả các tọa độ về dạng chuẩn hóa
    - Chuẩn hóa ảnh với mean và std của ImageNet

    :param image, tensor [C, H, W]
    :param bboxes, tensor [n_objects, 4]
    :param labels, tensor [n_objects]
    :param difficulties, tensor [n_objects]

    return:
    new_image, tensor [C, H, W]
    new_bboxes, tensor [new_n_objects, 4]
    new_labels, tensor [new_n_objects]
    new_difficulties, tensor [new_n_objects]
    """

    assert phase in {'train', 'test'}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_bboxes = bboxes
    new_labels = labels
    new_difficulties = difficulties

    if phase == 'train':
        new_image = photometric_distort(new_image)

        if random.random() < 0.5:
            new_image, new_bboxes = expand(new_image, bboxes, mean)

        new_image, new_bboxes, new_labels, new_difficulties = random_crop(new_image, new_bboxes, new_labels, new_difficulties)

        if random.random() < 0.5:
            new_image, new_bboxes = flip(new_image, new_bboxes)

    new_image, new_bboxes = resize(new_image, new_bboxes, dims=(300, 300))

    new_image = FT.normalize(new_image, mean, std)

    return new_image, new_bboxes, new_labels, new_difficulties