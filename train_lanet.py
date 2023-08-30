# Pre-train LA-Net
import torch
import torch.optim as optim 
import torch.nn as nn
from data import prepare_data
try:
    from net.model import get_model
except:
    from .net.model import get_model
import time, os
import fire
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from collections import OrderedDict
import re


def train(model, 
          model_save_path, 
          dataloader, 
          optimizer, 
          num_epochs, 
          device="cuda:0"):
    """Train Classifier"""
    best_acc = 0
    acc_history = []
    start_t = time.time() 
    max_save_count = 5
    save_counter = 0
    save_interval = 5
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    best_test_model = os.path.join(model_save_path, "best_model.pt") 
    mask_criterion = nn.BCELoss().to(device)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0
            running_corrects = 0
            for data in dataloader[phase]:
                inputs = data["image"].to(device)
                masks = data["mask"].to(device)
                mask_exist = data["mask_exist"].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    featmap_size = outputs[1].shape[-1]
                    batch_size = outputs[1].shape[0]
                    masks_inter = nn.functional.interpolate(masks, size=(featmap_size, featmap_size), mode="bilinear", align_corners=True)
                    mask_exist = mask_exist.view([batch_size, 1, 1, 1])
                    mask_pred = outputs[1] * mask_exist 
                    mask_loss = mask_criterion(mask_pred, masks_inter)
                    # get mask
                    mask_pred_cp = torch.clone(mask_pred)
                    mask_pred_cp[mask_pred >= 0.5] = 1
                    mask_pred_cp[mask_pred < 0.5] = 0
                    
                    if phase == "train":
                        mask_loss.backward()
                        optimizer.step()
                running_loss += mask_loss.item() * inputs.size(0)
                running_corrects += np.sum(mask_pred_cp.data.cpu().numpy() == masks_inter.data.cpu().numpy())
            datasize = len(dataloader[phase].dataset)
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / datasize / (featmap_size * featmap_size)
            print("{} Loss: {:.4f}; Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.lanet.state_dict(), best_test_model)
            if phase == "test":
                acc_history.append(epoch_acc)
        if not (epoch % save_interval):
            save_counter += 1
            torch.save(model.state_dict(), model_save_path+"/w_epoch_{}.pt".format(epoch+1))
            oldest = epoch+1-save_interval*max_save_count
            if os.path.exists(model_save_path+"/w_epoch_{}.pt".format(oldest)):
                os.remove(model_save_path+"/w_epoch_{}.pt".format(oldest))
    time_elapsed = time.time() - start_t 
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best val acc: {:.4f}".format(best_acc))
    # model.load_state_dict(torch.load(best_model_w, map_location=torch.device(device)))
    return model, acc_history

def load_weights(model, pretrained_weights, multi_gpu=False, device="cuda:0", num_classes=2):
    state_dict=torch.load(pretrained_weights, map_location=torch.device(device))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): # different class number, drop the last fc layer
        if re.search("fc", k):
            w_shape = list(v.size())
            # drop normal class weights if the number of classes of pretrained weights is different from current training setting.
            if w_shape != num_classes:
                w_shape = [num_classes, *w_shape[1:]]
                v = torch.rand(w_shape, requires_grad=True)
                if len(w_shape) > 2:
                    nn.init.kaiming_normal_(v) # weights
                else:
                    v.data.fill_(0.01) # bias
        if multi_gpu:
            name = k[7:] # remove 'module.' of dataparallel
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def run(model_name, 
        image_size=256, 
        num_classes=2, 
        batch_size=32, 
        num_epochs=40, 
        model_save_path="train_res", 
        device="cuda:0", 
        lr=0.001, 
        use_pretrained=True,
        pretrained_weights="",
        backbone_weights="",
        lanet_weights="",
        dataset="BUSI",
        num_gpus=1, 
        dilute_mask=0,
        use_cam=True,
        use_sam=True,
        use_mam=True, 
        map_size=14,
        reduction_ratio=16, 
        attention_kernel_size=3, 
        attention_num_conv=3):
    # get model 
    cbam_param = dict(use_cam=use_cam, 
                        use_sam=use_sam,
                        use_mam=use_mam,
                        reduction_ratio=reduction_ratio, 
                        attention_num_conv=attention_num_conv, 
                        attention_kernel_size=attention_kernel_size,
                        backbone_weights=backbone_weights,
                        lanet_weights=lanet_weights,
                        device=device)
    model = get_model(model_name=model_name,
                      num_classes=num_classes, 
                      use_pretrained=use_pretrained, 
                      return_logit=False,
                      map_size=map_size,
                      **cbam_param).to(device)
    # load pretrained model weights
    if pretrained_weights: 
        try:
            model = load_weights(model, pretrained_weights, multi_gpu=False, device=device, num_classes=num_classes)
        except:
            model = load_weights(model, pretrained_weights, multi_gpu=True, device=device, num_classes=num_classes)
    if dataset == "BUSI":
        train_file = "data/busi_train_binary.txt"
        test_file = "data/busi_test_binary.txt"
    elif dataset == "MAYO":
        train_file = "data/mayo_train_bbox.txt"
        test_file  = "data/mayo_test_bbox.txt"
    else:
        raise ValueError("Unknow dataset!")
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        # deploy model on multi-gpus
        model = nn.DataParallel(model, device_ids=device_ids)
    config = {"image_size": image_size, 
              "train": train_file, 
              "test": test_file, 
              "dataset": dataset,
              "mask": True,
              "dilute_mask": dilute_mask,
              }
    image_datasets, data_sizes = prepare_data(config)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=x=="train", batch_size=batch_size, num_workers=0,   drop_last=True) for x in ["train", "test"]}
    # optimizer
    print("optimized parameter names")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)
    print("-"*40)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model_ft, hist = train(model=model, 
                           model_save_path=model_save_path, 
                           dataloader=dataloaders, 
                           optimizer=optimizer, 
                           num_epochs=num_epochs, 
                           device=device)
    # torch.save(model_ft.state_dict(), model_save_path+'/best_model.pt')
    print("Val acc history: ", hist)

if __name__ == "__main__":
    fire.Fire(run)
