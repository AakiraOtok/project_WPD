from lib import *
from model import SSD
from VOC_utils import VOCUtils
from box_utils import Non_Maximum_Suppression, jaccard

def show(img, boxes, confs=None):
    """
    img có shape là [1, C, H, W] với C đang là RGB
    """
    img = img.squeeze(0).detach().permute(1, 2, 0)[:, :, (2, 1, 0)].cpu().numpy()
    for i in range(boxes.shape[0]):
        box = boxes[i]
        p1 = (int(box[0]*300), int(box[1]*300))
        p2 = (int(box[2]*300), int(box[3]*300))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 1)

        if confs is not None:
            conf = confs[i]
            cv2.putText(img, str(round(100*conf.item())), p1, 1, 1, (0, 0, 255), 1, 1)
            
    cv2.imwrite(r"H:\projectWPD\show_img.jpg", img)
    sys.exit()

def calc_APs(model, dataset, threshold=0.5):
    """
    tính APs và mAP của model, trả về 1 tensor gồm [nclass] phần tử
    """

    model.to("cuda")

    # Tạo test dataloader, xử lý từng bức ảnh (batch_size=1)
    #dataloader = VOC_create_test_dataloader(batch_size=1, shuffle=0, data_folder_path=r"H:\projectWPD\data", version=r"VOC2007", id_txt_file=r"test.txt")
    dboxes = model.create_prior_boxes().to("cuda")
    num_classes = 21

    # List chứa [nclass] list thành phần khác, mỗi list thành phần thứ i là các tensor TP/FP/confidence cho class thứ i
    TP     = [[] for _ in range(num_classes)]
    FP     = [[] for _ in range(num_classes)]
    confs  = [[] for _ in range(num_classes)]

    # Tensor [nclass] : tổng số ground truth box của mỗi class
    total  = torch.zeros(num_classes).to("cuda")

    with torch.set_grad_enabled(False):
        #for imgs, targets in tqdm(dataloader):
        for idx_dataset in tqdm(range(len(dataset))):
            img, bboxes, labels, difficult = dataset.__getitem__(idx_dataset)

            # Chuẩn bị data
            img    = img.unsqueeze(0).to("cuda")
            bboxes = bboxes.to("cuda")
            labels = labels.long().to("cuda")
            difficult = difficult.long().to("cuda")

            # Đưa vào mạng
            offset, conf = model(img)

            offset.squeeze_(0) # [1, 8732, 4]  ->  [8732, 4]
            conf.squeeze_(0)   # [1, 8732, 21] ->  [8732, 21]
            
            # [nbox, 4], [nbox]     ,  [nbox], nếu không có box được tìm thấy thì trả về None
            pred_bboxes, pred_labels, pred_confs = Non_Maximum_Suppression(dboxes, offset, conf, iou_threshold=0.45)

            # sort lại theo confidence score
            if (pred_bboxes != None):
                pred_confs, idx = torch.sort(pred_confs, descending=True)
                pred_bboxes     = pred_bboxes[idx]
                pred_labels     = pred_labels[idx]

            for cur_class in range(1, num_classes):
                # lọc các bboxes có có nhãn là class đang xét
                mask = (labels == cur_class)
                cur_bboxes = bboxes[mask]
                cur_difficult = difficult[mask]

                mask = (cur_difficult == 0)
                total[cur_class] += mask.sum()

                # lọc các predict boxes có có nhãn là class đang xét]
                if (pred_bboxes == None):
                    continue
                mask            = (pred_labels == cur_class)
                if (not mask.any()):
                    continue

                cur_pred_bboxes = pred_bboxes[mask]
                cur_pred_confs  = pred_confs[mask]

                cur_TP = torch.zeros(cur_pred_bboxes.size(0)).to("cuda")
                cur_FP = torch.zeros(cur_pred_bboxes.size(0)).to("cuda")
                matched = torch.LongTensor(cur_bboxes.size(0)).fill_(0).to("cuda")

                # Với mỗi pbox, tìm bbox overlap nhiều nhất, nếu max overlap < threshold hoặc bbox đó đã được match thì box đó là FP
                # ngược lại thì match bbox đó với pbox và pbox này là TP
                for pbox in range(cur_pred_bboxes.size(0)):
                    max_iou   = 0
                    best_bbox = 0

                    for bbox in range(cur_bboxes.size(0)):
                        overlap = jaccard(cur_pred_bboxes[pbox:pbox+1], cur_bboxes[bbox:bbox+1])[0, 0].item()
                        if overlap > max_iou:
                            max_iou   = overlap
                            best_bbox = bbox

                    if (max_iou > threshold):
                        if cur_difficult[best_bbox] == 0:
                            if matched[best_bbox] == 0:
                                cur_TP[pbox] = 1
                                matched[best_bbox] = 1
                            else:
                                cur_FP[pbox] = 1
                    else:
                        cur_FP[pbox] = 1

                TP[cur_class].append(cur_TP)
                FP[cur_class].append(cur_FP)
                confs[cur_class].append(cur_pred_confs)

        APs = torch.zeros(num_classes).to("cuda")

        # Tránh chia cho 0
        epsilon = 1e-6
    
        # Tính AP cho mỗi class
        for cur_class in range(1, num_classes):
            if len(TP[cur_class]) == 0:
                continue
            TP[cur_class] = torch.cat(TP[cur_class])
            FP[cur_class] = torch.cat(FP[cur_class])
            confs[cur_class] = torch.cat(confs[cur_class])

            confs[cur_class], idx = torch.sort(confs[cur_class], descending=True)
            TP[cur_class]         = TP[cur_class][idx]
            FP[cur_class]         = FP[cur_class][idx]

            TP_acc = torch.cumsum(TP[cur_class], dim=0)
            FP_acc = torch.cumsum(FP[cur_class], dim=0)

            precision = TP_acc/(TP_acc + FP_acc + epsilon)
            recall    = TP_acc/(total[cur_class] + epsilon)
            precision = torch.cat((torch.tensor([1]).to("cuda"), precision))
            recall    = torch.cat((torch.tensor([0]).to("cuda"), recall))

            APs[cur_class] = torch.trapz(precision, recall)
            
            print(torch.sum(TP[cur_class]))
            print(torch.sum(FP[cur_class]))
            plt.plot(recall.detach().cpu().numpy(), precision.detach().cpu().numpy())
            plt.show()

        return APs

if __name__ == "__main__":
    data_folder_path = r"H:\projectWPD\data"
    pretrain_path    = r"H:\projectWPD\VOC2007_trainval_checkpoint\iteration_120000.pth"
    
    model = SSD(pretrain_path, n_classes=21)
    dataset = VOCUtils(data_folder_path).make_dataset(r"VOC2007", r"test.txt", phase="valid")

    APs = calc_APs(model, dataset)
    APs = APs[1:] # bỏ background
    print(APs)
    print(APs.mean())