import pandas as pd
from stream import BIN_Data_Encoder
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
import torch
from torch.utils import data

import numpy as np
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
from torch.autograd import Variable

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

        label = Variable(torch.tensor(label).float()).to(device)

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to("cpu").numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    epsilon = 1e-8
    precision = tpr / (tpr + fpr + epsilon)


    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

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


config = BIN_config_DBPE()
BATCH_SIZE = config['batch_size']
params = {'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 6, 
            'drop_last': True}

data_path = 'pharos_full/FullPharos.csv'
df_test = pd.read_csv(data_path)
test_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test)
test_generator = data.DataLoader(test_set, **params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BIN_Interaction_Flat(**config).to(device)

# Load the state dictionary from the saved model
model.load_state_dict(torch.load("DAVIS_1_17.pth", map_location=device)) 

print("--- Go for Testing ---")
try:
    predictions_list = []
    with torch.set_grad_enabled(False):
        auc, auprc, f1, y_pred, loss, y_label = test(test_generator, model)
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
        predictions_df.to_csv("test_predictions_pharos_full.csv", index=False)
        print("Predictions saved to test_predictions_pharos_full.csv")
except Exception as e:
    print(f"Testing failed with error: {e}")
    
    
    
