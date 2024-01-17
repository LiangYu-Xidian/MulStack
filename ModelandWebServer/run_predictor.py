import torch
import joblib
import numpy as np
import os
import subprocess
import argparse
from multiprocessing import Process
from GetData import Mul_load_data, Get_Seq_Data
from GetLR import Calculate
from GetModel import TextCNN
import csv
device = torch.device("cpu")

def Nu_Get_Data(seq_list):

    data_list = []
    for i in seq_list:
        data = Get_Seq_Data('Nu', i, ['Kmer'], 3200, 3200, seq_list.index(i))
        data_list.append(data)
    return data_list

def Exo_Get_Data(seq_list):
    data_list = []
    for i in seq_list:
        data = Get_Seq_Data('Exo', i, ['Kmer'], 3200, 3200,seq_list.index(i))
        data_list.append(data)
    return data_list

def other_Get_Data(seq_list):
    data_list = []
    for i in seq_list:
        data = Get_Seq_Data('Cyto', i, ['Kmer'], 3000, 3000,seq_list.index(i))
        data_list.append(data)
    return data_list

def Meta(data,mRNA,w):

    if mRNA == 'Nu' or mRNA == 'Exo':
        lens=3200
    else:
        lens=3000

    dl_model = TextCNN(lens).to(device)
    model_name = './SaveModel/' + mRNA+ '.pth'
    dl_model.load_state_dict(torch.load(model_name))

    rf_model = joblib.load('./SaveModel/' + mRNA+ '.joblib')

    meta_label, meta_name = Calculate(dl_model, rf_model, data, w)
    return meta_label,meta_name
import sys
if __name__ == '__main__':

    # with open("./123.csv", 'w', newline='') as file:
    #     print("OK")
    #
    # exit()

    #uip = '/MulStack/webserver/temp/user/127.0.0.1_mRNA_1673154787.9017649/'
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-uip', type=str)
    # args = parser.parse_args()
    # uip = args.uip
    uip = './temp/user/127.0.0.1_mRNA_1673174416.3199015/'
    csv_name = uip + 'result.csv'
    uip = uip + 'human_mRNA.fasta'
    seq_list = []
    name_list = []

    for i in open(uip):
        if i[0] != '>':
            seq_list.append(i.strip())

    Nu_list = Nu_Get_Data(seq_list)
    Exo_list = Exo_Get_Data(seq_list)
    Other_list = other_Get_Data(seq_list)

    Nu_label,Nu_name = Meta(Nu_list,'Nu',0)
    Cyto_label, Cyto_name = Meta(Other_list,'Cyto', 0)
    ER_label, ER_name = Meta(Other_list,'ER', 0)
    Exo_label, Exo_name = Meta(Exo_list,'Exo', 0)
    Mem_label, Mem_name = Meta(Other_list,'Mem', 0)
    Ribo_label, Ribo_name = Meta(Other_list,'Ribo',0)


    size = len(Nu_label)
    with open(csv_name,'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Nucleus","Exosome","Cytosol","Ribosome","Membrane","ER"])
        for i in range(size):
            if Nu_label[Nu_name.index(i)] == '1' : nu_label = "True"
            else: nu_label = "False"
            if Cyto_label[Cyto_name.index(i)] == '1' : cyto_label = "True"
            else: cyto_label = "False"
            if ER_label[ER_name.index(i)] == '1' : er_label = "True"
            else: er_label = "False"
            if Exo_label[Exo_name.index(i)] == '1' : exo_label = "True"
            else: exo_label = "False"
            if Mem_label[Mem_name.index(i)] == '1' : mem_label = "True"
            else: mem_label = "False"
            if Ribo_label[Ribo_name.index(i)] == '1' : ribo_label = "True"
            else: ribo_label = "False"

            writer.writerow([str(nu_label),str(exo_label),str(cyto_label),str(ribo_label),str(mem_label),str(er_label)])