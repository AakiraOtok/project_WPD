from utils.lib import *
from utils.VOC_utils import VOCUtils, collate_fn
from model import SSD
from utils.box_utils import MultiBoxLoss

def train(dataloader, model, criterion, optimizer):

    model.to("cuda")
    dboxes = model.create_prior_boxes().to("cuda")
    iteration = -1
    while(1):
        for batch_images, batch_bboxes, batch_labels, batch_difficulties in dataloader: 
            iteration += 1
            t_batch = time.time()

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
                # save lại mỗi 1000 iteration
            if (iteration + 1) % 1000 == 0:
                torch.save(model.state_dict(), r"H:\projectWPD\VOC2007_trainval_checkpoint\iteration_" + str(iteration + 1) + ".pth")
                print("Saved model at iteration : {}".format(iteration + 1))


if __name__ == "__main__":
    data_folder_path = r"H:\projectWPD\data"

    dataloader = VOCUtils(data_folder_path).make_dataloader(r"VOC2007", r"trainval.txt", 32, True, collate_fn, phase='train',num_worker=4, pin_memory=True)
    model      = SSD(n_classes=21)
    criterion  = MultiBoxLoss(num_classes=21)
    optimizer  = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    train(dataloader, model, criterion, optimizer)