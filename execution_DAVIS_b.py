import copy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import traceback
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

torch.manual_seed(1)  # reproducible torch:2 np:3
np.random.seed(1)

import torch

from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

print(torch.cuda.is_available())

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(
            d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda()
        )

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to("cpu").numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    print(fpr)
    print(tpr)
    print(thresholds)
    print("Unique predictions:", np.unique(y_pred))


    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    print(f1)
        
    if len(f1) > 5:
        thred_optim = thresholds[5:][np.argmax(f1[5:])]
    else:
        thred_optim = thresholds[np.argmax(f1)]  # Use the max F1 from available values


    # thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print("Confusion Matrix : \n", cm1)
    print("Recall : ", recall_score(y_label, y_pred_s))
    print("Precision : ", precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print("Accuracy : ", accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print("Sensitivity : ", sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print("Specificity : ", specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return (
        roc_auc_score(y_label, y_pred),
        average_precision_score(y_label, y_pred),
        f1_score(y_label, outputs),
        y_pred,
        loss.item(),
        y_label
    )


def main(fold_n, lr):
    config = BIN_config_DBPE()

    lr = lr
    BATCH_SIZE = config["batch_size"]
    train_epoch = 100

    loss_history = []

    model = BIN_Interaction_Flat(**config)

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # opt = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)

    print("--- Data Preparation ---")

    params = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 6,
        "drop_last": True,
    }

    dataFolder = "./dataset/DAVIS_b3"
    df_train = pd.read_csv(dataFolder + "/train.csv")
    df_val = pd.read_csv(dataFolder + "/val.csv")
    df_test = pd.read_csv(dataFolder + "/test.csv")

    training_set = BIN_Data_Encoder(
        df_train.index.values, df_train.Label.values, df_train
    )
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val.Label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    print("--- Go for Training ---")
    torch.backends.cudnn.benchmark = True
    for epo in range(train_epoch):
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(
                d.long().cuda(),
                p.long().cuda(),
                d_mask.long().cuda(),
                p_mask.long().cuda(),
            )

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss = loss_fct(n, label)
            loss_history.append(loss)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                print(
                    "Training at Epoch "
                    + str(epo + 1)
                    + " iteration "
                    + str(i)
                    + " with loss "
                    + str(loss.cpu().detach().numpy())
                )

        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss, _ = test(validation_generator, model)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc

            print(
                "Validation at Epoch "
                + str(epo + 1)
                + " , AUROC: "
                + str(auc)
                + " , AUPRC: "
                + str(auprc)
                + " , F1: "
                + str(f1)
            )

    print("--- Go for Testing ---")
    try:
        predictions_list = []
        with torch.set_grad_enabled(False):
            auc, auprc, f1, y_pred, loss, y_label = test(testing_generator, model_max)
            print(
                "Testing AUROC: "
                + str(auc)
                + " , AUPRC: "
                + str(auprc)
                + " , F1: "
                + str(f1)
                + " , Test loss: "
                + str(loss)
            )
            # Save predictions to CSV
            predictions_df = pd.DataFrame({"True_Label": y_label, "Predicted_Score": y_pred})
            predictions_df.to_csv("test_predictions_b3.csv", index=False)
            print("Predictions saved to test_predictions_b3.csv")
    except Exception as e:
        print(f"Testing failed due to: {e}")
        traceback.print_exc()
    return model_max, loss_history


# fold 1
# biosnap interaction times 1e-6, flat, batch size 64, len 205, channel 3, epoch 50
s = time()
model_max, loss_history = main(1, 5e-6)
torch.save(model_max.state_dict(), "DAVIS_b_3.pth")
e = time()
print(e - s)
