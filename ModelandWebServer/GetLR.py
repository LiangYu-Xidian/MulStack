import os, random, math, time, sys
import multiprocessing as mul
from itertools import combinations
import pickle

from sklearn.linear_model import LogisticRegression


from GetPerformance import performance

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from GetCrossValidation import soft
import torch.utils.data as Data
import torch
import numpy

from util import list2libsvm
import numpy as np

penalty="l1"
solver="saga"


def Calculate(dl_model, rf_model, seq_list, w):

    process_name = []
    process_seq = []
    process_ml = []

    for data in seq_list:
        process_seq.append(data.seq)
        process_ml.append(data.mlfeature)
        process_name.append(data.name)


    model = dl_model.eval()
    process_batch, process_ml_batch, process_name_batch= \
        torch.tensor(process_seq,dtype=torch.float32).long(),  torch.tensor(process_ml, dtype=torch.float32),torch.tensor(process_name, dtype=torch.float32)
    process_dataset = Data.TensorDataset(process_batch,process_ml_batch,process_name_batch)
    process_loader = torch.utils.data.DataLoader(
        process_dataset,
        batch_size=32,
    )

    meta_label = []
    meta_name = []

    with torch.no_grad():
        for process_x,process_ml,process_name in process_loader:
            # process_x = process_x.cuda()
            # process_ml = process_ml.cuda()

            result = model(process_x,process_ml).data
            dl_pro_val = []
            for data in result:
                probability = soft(data).cpu().numpy()
                p = probability[1]*w
                temp = []
                temp.append(p)
                dl_pro_val.append(temp)

            rf_ml = process_ml.cpu().numpy()
            rf_p_val = rf_model.predict_proba(rf_ml)
            rf_pro_val = []
            for val in rf_p_val:
                p = val[1]*(1-w)
                val_list = []
                val_list.append(p)
                rf_pro_val.append(val_list)

            dl_val = numpy.array(dl_pro_val)
            ml_val = numpy.array(rf_pro_val)

            result = dl_val+ml_val
            result = result.squeeze(1).tolist()

            for s in range(len(result)):

                if result[s] >= 0.5:
                    meta_label.append('1')
                else:
                    meta_label.append('0')

            for name in process_name.numpy():
                meta_name.append(name)

        return meta_label,meta_name

