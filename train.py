import torch
import torch.optim as optim 
import torch.nn as nn
from data import prepare_data
from model import get_model
import copy 
import time, os
import fire
import numpy as np


def train(model, 
         model_name, 
         model_save_path, 
         dataloader, 
         optimizer, 
         criterion, 
         num_epochs, 
         input_size, 
         device="cpu"):
    """Train Classifier"""
    # best_model_w = copy.deepcopy(model.state_dict())
    best_acc = 0
    acc_history = []
    start_t = time.time() 
    # add log to tensorborad 
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    best_test_model = os.path.join(model_save_path, "best_model.pt") 
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
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # if use_cent_loss:
                #     optimizer_cent.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    if (model_name == "inception_v3") and (phase == "train"): 
                        outputs, aux = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs[0], labels)
                        # print(outputs[0], labels)
                    _, preds = torch.max(outputs[0], 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                # print(preds, labels.data, torch.sum(preds == labels.data))
                running_corrects += torch.sum(preds == labels.data)
            datasize = len(dataloader[phase].dataset)
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects.double() / datasize
            print("{} Loss: {:.4f}, Acc{:.4f}".format(phase, epoch_loss, epoch_acc))
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_test_model)
            if phase == "test":
                acc_history.append(epoch_acc)
        if not (epoch % 5):
            torch.save(model.state_dict(), model_save_path+"/w_epoch_{}.pt".format(epoch))
    time_elapsed = time.time() - start_t 
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    print("Best val acc: {:.4f}".format(best_acc))
    # model.load_state_dict(torch.load(best_model_w, map_location=torch.device(device)))
    return model, acc_history


def run(model_name, 
        input_size=448, 
        num_classes=3, 
        batch_size=32, 
        num_epochs=40, 
        model_save_path="train_res", 
        device="cuda:0", 
        lr=0.001, 
        moment=0.9, 
        use_pretrained=True,
        dataset="BUSI",
        num_gpus=1):
    # get model 
    model = get_model(model_name=model_name,
                      num_classes=num_classes, 
                      use_pretrained=use_pretrained, return_logit=False).to(device)
    if dataset == "BUSI":
        train_file = "train_sample.txt"
        test_file = "test_sample.txt"
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        # deploy model on multi-gpus
        model = nn.DataParallel(model, device_ids=device_ids)
    config = {"input_size": input_size, "train": train_file, "test": test_file, "dataset": dataset}
    # config = {"input_size": input_size, "train": "train_sample.txt", "test": "train_sample.txt", "dataset": dataset}
    image_datasets, data_sizes = prepare_data(config)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=x=="train", batch_size=batch_size, num_workers=4,   drop_last=True) for x in ["train", "test"]}
    # loss function
    # cls_weight = [2.0, 1.0, 1.0]
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer
    print("optimized parameter names")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)
    print("-"*10)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=moment) 
    # if use_cent_loss:
    #     criterion_cent = CenterLoss(num_classes, feat_dim=feat_dim).to(device)
    #     optim_cent = torch.optim.SGD(criterion_cent.parameters(), lr=lr_cent)
    # else:
    #     criterion_cent, optim_cent = None, None
    model_ft, hist = train(model=model, 
                           model_name=model_name, 
                           model_save_path=model_save_path, 
                           dataloader=dataloaders, 
                           optimizer=optimizer, 
                           criterion=criterion, 
                           num_epochs=num_epochs, 
                           input_size=input_size, 
                           device=device)
    # torch.save(model_ft.state_dict(), model_save_path+'/best_model.pt')
    print("Val acc history: ", hist)

if __name__ == "__main__":
    fire.Fire(run)
    # # training config
    # input_size = 224
    # num_classes = 3 
    # batch_size = 16
    # num_epoches = 40
    # model_name = "resnet50"
    # device = "cuda:0"
    # input_dir = "/shared/anastasio5/COVID19/data/covidx"
    # model_save_path = "covidx_res50"

