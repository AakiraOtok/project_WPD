from utils.lib import *
from utils.VOC_utils import VOCUtils, collate_fn
from utils.COCO_utils import COCOUtils, COCO_collate_fn
from model.SSD300 import SSD300
from model.SSD512 import SSD512
from model.FPN_SSD300 import FPN_SSD300
from model.FPN_SSD512 import FPN_SSD512
from utils.box_utils import MultiBoxLoss
from utils.augmentations_utils import CustomAugmentation

def train_model(dataloader, model, criterion, optimizer, adjustlr_schedule=(80000, 100000), max_iter=200000):
    torch.backends.cudnn.benchmark = True
    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    iteration = -1

    while(1):
        for batch_images, batch_bboxes, batch_labels, batch_difficulties in dataloader: 
            iteration += 1
            t_batch = time.time()
            if iteration in adjustlr_schedule:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

            batch_size   = batch_images.shape[0]
            batch_images = batch_images.to("cuda")
            for idx in range(batch_size):
                batch_bboxes[idx]       = batch_bboxes[idx].to("cuda")
                batch_labels[idx]       = batch_labels[idx].to("cuda")
                batch_difficulties[idx] = batch_difficulties[idx].to("cuda")

            loc, conf = model(batch_images)

            loss = criterion(loc, conf, dboxes, batch_bboxes, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()

            print("iteration : {}, time = {}, loss = {}".format(iteration + 1, round(time.time() - t_batch, 2), loss))
                # save lại mỗi 10000 iteration
            if (iteration + 1) % 10000 == 0:
                torch.save(model.state_dict(), r"H:\projectWPD\VOC_checkpoint\iteration_" + str(iteration + 1) + ".pth")
                print("Saved model at iteration : {}".format(iteration + 1))
                if iteration + 1 == max_iter:
                    sys.exit()

def train_on_COCO(size=300, version = "original", pretrain_path=None):
    train_folder_path  = r"H:\data\COCO\train2014"
    val35k_folder_path = r"H:\data\COCO\val2014"
    train_file         = r"H:\data\COCO\instances_train2014.json"
    val35k_file        = r"H:\data\COCO\instances_valminusminival2014.json"
 
    train      = COCOUtils(train_folder_path, train_file).make_dataset(phase="train", transform=CustomAugmentation(size=size))
    val35k     = COCOUtils(val35k_folder_path, val35k_file).make_dataset(phase="train", transform=CustomAugmentation(size=size))
    dataset    = data.ConcatDataset([train, val35k])
    dataloader = data.DataLoader(dataset, 32, True, collate_fn=COCO_collate_fn, num_workers=6, pin_memory=True)

    if version == "original":
        if size==300:
            model = SSD300(data_train_on="COCO", n_classes=81, pretrain_path=pretrain_path)
        elif size==512:
            model = SSD512(data_train_on="COCO", n_classes=81, pretrain_path=pretrain_path)
    elif version == "FPN":
        if size == 300:
            model = FPN_SSD300(data_train_on="COCO", n_classes=81, pretrain_path=pretrain_path)
        elif size == 512:
            model = FPN_SSD512(data_train_on="COCO", n_classes=81, pretrain_path=pretrain_path)

    criterion  = MultiBoxLoss(num_classes=81)

    return dataloader, model, criterion

def train_on_VOC(size=300, version="original", pretrain_path=None):
    data_folder_path = r"H:\projectWPD\data"
    voc              = VOCUtils(data_folder_path)
    
    dataset1         = voc.make_dataset(r"VOC2007", r"trainval.txt", transform=CustomAugmentation(size=size))
    dataset2         = voc.make_dataset(r"VOC2012", r"trainval.txt", transform=CustomAugmentation(size=size))
    dataset          = data.ConcatDataset([dataset1, dataset2])

    dataloader       = data.DataLoader(dataset, 32, True, num_workers=6, collate_fn=collate_fn, pin_memory=True)

    if version == "original":
        if size==300:
            model = SSD300(n_classes=21, pretrain_path=pretrain_path)
        elif size==512:
            model = SSD512(n_classes=21, pretrain_path=pretrain_path)
    elif version == "FPN":
        if size == 300:
            model = FPN_SSD300(n_classes=21, pretrain_path=pretrain_path)
        elif size == 512:
            model = FPN_SSD512(n_classes=21, pretrain_path=pretrain_path)

    #criterion  = MultiBoxLoss(num_classes=21)
    from utils.box_utils import MultiBox_Focal_Loss
    criterion  = MultiBox_Focal_Loss(num_classes=21)

    return dataloader, model, criterion


if __name__ == "__main__":

    dataloader, model, criterion = train_on_VOC(version="FPN", size=300)
    #dataloader, model, criterion = train_on_COCO()

    biases     = []
    not_biases = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer  = optim.SGD(params=[{'params' : biases, 'lr' : 2 * 1e-3}, {'params' : not_biases}], lr=1e-3, momentum=0.9, weight_decay=5e-4)

    train_model(dataloader, model, criterion, optimizer, adjustlr_schedule=(80000, 100000), max_iter=120000)
