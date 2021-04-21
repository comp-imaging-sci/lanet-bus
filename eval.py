import torch 
from model import get_model
from data import prepare_data
import fire
import numpy as np
from scipy import stats
from collections import OrderedDict

BUSI_LABELS = ["normal", "malignant", "benign"]

def infer_model(model_name, 
                num_classes, 
                model_weights,  
                input_size=224, 
                device="cpu", 
                dataset="BUSI"):
    model = get_model(model_name=model_name, num_classes=num_classes, use_pretrained=False, return_logit=False).to(device)
    state_dict=torch.load(model_weights, map_location=torch.device(device))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    #new_state_dict = state_dict
    model.load_state_dict(new_state_dict)
    model.eval()
    # config = {"input_size": input_size, "train": "train_sample.txt", "test": "train_sample.txt"}
    if dataset == "BUSI":
        train_file = "train_sample.txt"
        test_file = "test_sample.txt"
    config = {"input_size": input_size, "train": train_file, "test": test_file, "dataset": dataset}
    image_datasets, data_sizes = prepare_data(config)
    dataloader = torch.utils.data.DataLoader(image_datasets["test"], shuffle=False)  

    # result matrics:
    #      __________________________________________
    #      | gt \ pred | Normal | COVID | Pneumonia | 
    #      ------------------------------------------
    #      |  Normal   |        |       |           |
    #      |  COVID    |        |       |           |
    #      |  Pneumonia|        |       |           |
    #      ------------------------------------------
    if dataset == "BUSI":
        result_matrics = np.zeros((3, 3))
    with torch.no_grad():
        for img, label in dataloader:
            tag = label.numpy()[0]
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            _, pred = torch.max(outputs[0], 1)
            pred = int(pred.cpu().numpy()[0])
            result_matrics[tag][pred] += 1
    return result_matrics

def mean_confidence_interval(x, confidence=0.95):
    # get CI with 0.95 confidence following normal gaussian distribution
    n = len(x)
    m, se = np.mean(x), stats.sem(x)
    ci = stats.t.ppf((1 + confidence) / 2., n-1) * se
    # ci = 1.96 * se  # assume gaussian distribution
    return m, ci

def eval(model_name, 
         num_classes, 
         model_weights,  
         input_size=224, 
         device="cpu", 
         use_arc_loss=False, 
         arc_weights=None, 
         use_patch_input=False,
         num_patches=1, 
         patch_style="random",
         max_size=1680,
         dataset="covidx"):
    """
    Output metrics of model, including: Precision, Sensitivity, AUC
    """
    result_matrics = infer_model(model_name, 
                                num_classes, 
                                model_weights, 
                                input_size=input_size, 
                                device=device, 
                                dataset=dataset)
    # precision: TP / (TP + FP)
    print("result matrics: ", result_matrics)
    # res_acc = [result_matrics[i, i]/np.sum(result_matrics[:,i]) for i in range(num_classes)]
    res_acc = []
    # sensitivity: TP / (TP + FN)
    res_sens = []
    # res_sens = [result_matrics[i, i]/np.sum(result_matrics[i,:]) for i in range(num_classes)]
    # specificity: TN / (TN+FP)
    res_speci = []
    # f1 score: 2TP/(2TP+FP+FN)
    f1_score = []
    for i in range(num_classes):
        TP = result_matrics[i,i]
        FN = np.sum(result_matrics[i,:])-TP
        spe_matrics = np.delete(result_matrics, i, 0)
        FP = np.sum(spe_matrics[:, i])
        TN = np.sum(spe_matrics) - FP
        acc = TP/(TP+FP)
        sens = TP/(TP+FN)
        speci = TN/(TN+FP)
        f1 = 2*TP/(2*TP+FP+FN)
        res_acc.append(acc)
        res_speci.append(speci)
        res_sens.append(sens)
        f1_score.append(f1)
    if dataset == "BUSI":
        print('Precision: Normal: {0:.3f}, malignant: {1:.3f}, benign: {2:.3f}, avg: {3:.3f}'.format(res_acc[0],res_acc[1],res_acc[2], np.mean(res_acc)))
        print('Sensitivity: Normal: {0:.3f}, malignant: {1:.3f}, benign: {2:.3f}, avg: {3:.3f}'.format(res_sens[0],res_sens[1],res_sens[2], np.mean(res_sens)))
        print('Specificity: Normal: {0:.3f}, malignant: {1:.3f}, benign: {2:.3f}, avg: {3:.3f}'.format(res_speci[0],res_speci[1],res_speci[2], np.mean(res_speci)))
        print('F1 score: Normal: {0:.3f}, malignant: {1:.3f}, benign: {2:.3f}, avg: {3:.3f}'.format(f1_score[0],f1_score[1],f1_score[2], np.mean(f1_score)))          
    else:
        print("unknown dataset")


if __name__ == "__main__":
    fire.Fire(eval)
