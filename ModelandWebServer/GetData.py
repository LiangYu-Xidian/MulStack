# -*- coding:utf-8 -*-
import multiprocessing as mp
import sys
import Kmer, DAC, PseDNC

mRNAList = ["Nu", "Exo", "Cyto", "Cytoplasm", "Ribo", "Mem", "ER"]
src_vocab = {'P': 0, 'A': 1, 'G': 2, 'C': 3, 'U': 4}


#用于存储序列的机器学习特征和标签以及转换的序列
class MyDataset(object):
    def __init__(self, seq, mlfeature,name):
        super(MyDataset, self).__init__()
        self.seq = seq
        self.mlfeature = mlfeature
        self.name = name


#①加载Nu亚细胞的特征
'''
[Nu_Kmer k=5]
[Nu_DAC lag=6] [Nu_DCC lag=5] [Nu_DACC lag=7]
[Nu_PC w=0.2] [Nu_SC w=0.5]
[Nu_Mismatch k=4 m=1]  [Nu_MAC lamada=6] [Nu_NMBAC lamada=6] [Nu_Subsequence 0.1]
'''
def load_Nu_ml_feature(mRNA,seq,Feature_list):
    if mRNA != 'Nu':
        print("Wrong type,we need Nu")
        exit()
    '''
    #Nu: DACC、Kmer、DCC、SC、PC、DAC
    '''

    feature_vector = []
    for feature in Feature_list:
        if feature == 'Kmer':   #Nu_Kmer k=4
            Kmerres = Kmer.make_kmer_vector(k=3, seq=seq)
            feature_vector = feature_vector + Kmerres

        elif feature == 'DCC':  #Nu_DCC lag=5
            DCCres = DAC.make_DAC_vector(seq=seq, method='DCC', lag=1, lamada=None)
            feature_vector = feature_vector + DCCres

        elif feature == 'DAC':  #Nu_DAC lag=6
            DACres = DAC.make_DAC_vector(seq=seq, method='DAC', lag=4, lamada=None)
            feature_vector = feature_vector + DACres

        elif feature == 'DACC':     #Nu_DACC lag=7
            DACCres = DAC.make_DAC_vector(seq=seq, method='DACC', lag=1, lamada=None)
            feature_vector = feature_vector + DACCres

        elif feature == 'PC':   #Nu_PC w=0.2
            PCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='PC-PseDNC-General', k=None, w=0.2, lamada=1)
            feature_vector = feature_vector + PCPseDNCres

        elif feature == 'SC':   #Nu_SC w=0.5
            SCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='SC-PseDNC-General', k=None, w=0.2, lamada=1)
            feature_vector = feature_vector + SCPseDNCres

    return feature_vector

#②加载Exo亚细胞特征
'''
[Exo_Kmer k=5]
[Exo_DAC lag=7] [Exo_DCC lag=2] [Exo_DACC lag=2]
[Exo_PC w=0.4] [Exo_SC w=0.2]
[Exo_Mismatch k=4 m=2]  [Exo_MAC lamada=7] [Exo_NMBAC lamada=6] [Exo_Subsequence 0.1]
'''
def load_Exo_ml_feature(mRNA,seq,Feature_list):
    if mRNA != 'Exo':
        print("Wrong type,we need Exo")
        exit()

    '''
    Exo: DACC、Kmer、SC、PC、DCC、DAC
    '''
    feature_vector = []
    for feature in Feature_list:

        if feature == 'Kmer':   #Exo_Kmer k=5
            Kmerres = Kmer.make_kmer_vector(k=3, seq=seq)
            feature_vector = feature_vector + Kmerres

        elif feature == 'DCC':  #Exo_DCC lag=2
            DCCres = DAC.make_DAC_vector(seq=seq, method='DCC', lag=2, lamada=None)
            feature_vector = feature_vector + DCCres

        elif feature == 'DAC':  #Exo_DAC lag=7
            DACres = DAC.make_DAC_vector(seq=seq, method='DAC', lag=7, lamada=None)
            feature_vector = feature_vector + DACres

        elif feature == 'DACC':     #Exo_DACC lag=2
            DACCres = DAC.make_DAC_vector(seq=seq, method='DACC', lag=2, lamada=None)
            feature_vector = feature_vector + DACCres

        elif feature == 'PC':   #Exo_PC w=0.4
            PCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='PC-PseDNC-General', k=None, w=0.4, lamada=1)
            feature_vector = feature_vector + PCPseDNCres

        elif feature == 'SC':   #Exo_SC w=0.2
            SCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='SC-PseDNC-General', k=None, w=0.2, lamada=1)
            feature_vector = feature_vector + SCPseDNCres

    return feature_vector

#③加载Cytoplasm亚细胞特征
'''
[Cyto_Kmer k=5]
[Cyto_DAC lag=7] [Cyto_DCC lag=5] [Cyto_DACC lag=7]
[Cyto_PC w=0.7] [Cyto_SC w=0.6]
[Cyto_Mismatch k=4 m=3]  [Cyto_MAC lamada=4] [Cyto_NMBAC lamada=1] [Cyto_Subsequence 0.1]
'''
def load_Cyto_ml_feature(mRNA,seq,Feature_list):
    if mRNA != 'Cyto':
        print("Wrong type,we need Cyto")
        exit()

    '''
    Cyto: DACC、Kmer、DCC、SC、PC、DAC
    '''
    feature_vector = []
    for feature in Feature_list:

        if feature == 'Kmer':   #Cyto_Kmer k=4
            Kmerres = Kmer.make_kmer_vector(k=3, seq=seq)
            feature_vector = feature_vector + Kmerres

        elif feature == 'DCC':  #Cyto_DCC lag=5
            DCCres = DAC.make_DAC_vector(seq=seq, method='DCC', lag=5, lamada=None)
            feature_vector = feature_vector + DCCres

        elif feature == 'DAC':  #Cyto_DAC lag=7
            DACres = DAC.make_DAC_vector(seq=seq, method='DAC', lag=5, lamada=None)
            feature_vector = feature_vector + DACres

        elif feature == 'DACC':     #Cyto_DACC lag=7
            DACCres = DAC.make_DAC_vector(seq=seq, method='DACC', lag=7, lamada=None)
            feature_vector = feature_vector + DACCres

        elif feature == 'PC':   #Cyto_PC w=0.7
            PCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='PC-PseDNC-General', k=None, w=0.7, lamada=1)
            feature_vector = feature_vector + PCPseDNCres

        elif feature == 'SC':   #Cyto_SC w=0.6
            SCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='SC-PseDNC-General', k=None, w=0.6, lamada=1)
            feature_vector = feature_vector + SCPseDNCres

    return feature_vector

#④加载Ribosome亚细胞特征
def load_Ribo_ml_feature(mRNA, seq, Feature_list):
    if mRNA != 'Ribo':
        print("Wrong type,we need Ribo")
        exit()

    feature_vector = []

    for feature in Feature_list:

        if feature == 'Kmer':
            Kmerres = Kmer.make_kmer_vector(k=3, seq=seq) #Kmer k = 4
            feature_vector = feature_vector + Kmerres

        elif feature == 'SC':
            SCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='SC-PseDNC-General', k=None, w=0.2, lamada=1)
            feature_vector = feature_vector + SCPseDNCres

        elif feature == 'PC':
            PCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='PC-PseDNC-General', k=None, w=0.2, lamada=1)
            feature_vector = feature_vector + PCPseDNCres
        elif feature == 'DCC':
            DCCres = DAC.make_DAC_vector(seq=seq, method='DCC', lag=5, lamada=None)
            feature_vector = feature_vector + DCCres
        elif feature == 'DACC':
            DACCres = DAC.make_DAC_vector(seq=seq, method='DACC', lag=3, lamada=None)
            feature_vector = feature_vector + DACCres
        elif feature == 'DAC':
            DACCres = DAC.make_DAC_vector(seq=seq, method='DACC', lag=3, lamada=None)
            feature_vector = feature_vector + DACCres

    return feature_vector

#⑤加载Mem亚细胞特征
'''
[Mem_Kmer k=5]
[Mem_DAC lag=3] [Mem_DCC lag=1] [Mem_DACC lag=2]
[Mem_PC w=0.8] [Mem_SC w=0.8]
[Mem_Mismatch k=4 m=2]  [Mem_MAC lamada=7] [Mem_NMBAC lamada=6] [Mem_Subsequence 0.1]
'''
def load_Mem_ml_feature(mRNA,seq,Feature_list):
    if mRNA != 'Mem':
        print("Wrong type,we need Mem")
        exit()

    '''
    Mem: DACC、DAC、SC、DCC、Kmer、PC
    '''
    feature_vector = []
    for feature in Feature_list:

        if feature == 'Kmer':   #Mem_Kmer k=4
            Kmerres = Kmer.make_kmer_vector(k=3, seq=seq)
            feature_vector = feature_vector + Kmerres

        elif feature == 'DCC':  #Mem_DCC lag=1
            DCCres = DAC.make_DAC_vector(seq=seq, method='DCC', lag=1, lamada=None)
            feature_vector = feature_vector + DCCres

        elif feature == 'DAC':  #Mem_DAC lag=3
            DACres = DAC.make_DAC_vector(seq=seq, method='DAC', lag=3, lamada=None)
            feature_vector = feature_vector + DACres

        elif feature == 'DACC':     #Mem_DACC lag=2
            DACCres = DAC.make_DAC_vector(seq=seq, method='DACC', lag=2, lamada=None)
            feature_vector = feature_vector + DACCres

        elif feature == 'PC':   #Mem_PC w=0.8
            PCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='PC-PseDNC-General', k=None, w=0.8, lamada=1)
            feature_vector = feature_vector + PCPseDNCres

        elif feature == 'SC':   #Mem_SC w=0.8
            SCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='SC-PseDNC-General', k=None, w=0.8, lamada=1)
            feature_vector = feature_vector + SCPseDNCres

    return feature_vector

#⑥加载ER亚细胞的特征
def load_ER_ml_feature(mRNA, seq, Feature_list):
    if mRNA != 'ER':
        print("Wrong type,we need ER")
        exit()

    feature_vector = []
    for feature in Feature_list:
        if feature == 'PC':
            PCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='PC-PseDNC-General', k=None, w=0.8, lamada=1)
            feature_vector = feature_vector + PCPseDNCres

        elif feature == 'SC':
            PCPseDNCres = PseDNC.getPCPseDNC(seq=seq, alphabet='RNA', method='SC-PseDNC-General', k=None, w=0.8, lamada=1)
            feature_vector = feature_vector + PCPseDNCres

        elif feature == 'DCC':
            DCCres = DAC.make_DAC_vector(seq=seq, method='DCC', lag=4, lamada=None)
            feature_vector = feature_vector + DCCres

        elif feature == 'Kmer':
            Kmerres = Kmer.make_kmer_vector(k=3, seq=seq)
            feature_vector = feature_vector + Kmerres

        elif feature == 'DAC':
            DACres = DAC.make_DAC_vector(seq=seq, method='DAC', lag=3, lamada=None)
            feature_vector = feature_vector + DACres

        elif feature == 'DACC':
            DACCres = DAC.make_DAC_vector(seq=seq, method='DACC', lag=2, lamada=None)
            feature_vector = feature_vector + DACCres

    return feature_vector


#mRNAList = ["Nu", "Exo", "Cyto", "Cytoplasm", "Ribo", "Mem", "ER"]
#计算序列的核苷酸特征
def Get_ML_Feature(mRNA,seq,Feature):

    if mRNA == 'Nu':
        ml_feature = load_Nu_ml_feature('Nu', seq, Feature)
    elif mRNA == 'Exo':
        ml_feature = load_Exo_ml_feature('Exo', seq, Feature)
    elif mRNA == 'Cyto':
        ml_feature = load_Cyto_ml_feature('Cyto', seq, Feature)
    elif mRNA == 'Ribo':
        ml_feature = load_Ribo_ml_feature('Ribo', seq, Feature)
    elif mRNA == 'Mem':
        ml_feature = load_Mem_ml_feature('Mem', seq, Feature)
    elif mRNA == 'ER':
        ml_feature = load_ER_ml_feature('ER', seq, Feature)

    return ml_feature

#处理序列的信息
def Get_Pro_Seq(seq,left,right):

    seq = seq.strip()
    if len(seq) >= left + right:
        seq_left = seq[: left]
        seq_right = seq[-right:]
        pos_seq = seq_left + seq_right
    else:
        pos_seq = seq.ljust(left + right, 'P')

    num_data = []
    # 用数字编码代替seq
    for n in pos_seq:
        num = src_vocab[n]
        num_data.append(num)
    return num_data

#将核苷酸特征和序列信息特征结合
def Get_Seq_Data(mRNA,seq,Feature,left,right,name):

    ml = Get_ML_Feature(mRNA=mRNA, seq=seq, Feature=Feature)
    numdata = Get_Pro_Seq(seq=seq, left=left, right=right)
    data = MyDataset(numdata, ml, name)
    return data


#多线程处理序列
class Mul_load_data(object):

    def __init__(self, filename, mRNA, Feature, left, right, cpucore):
        self.filename = filename
        self.seq_list = mp.Manager().list()
        self.data_list = mp.Manager().list()

        self.mRNA = mRNA
        self.Feature = Feature
        self.left = left
        self.right = right
        self.cpucore = cpucore

        self.members = [i for i in range(self.cpucore)]


    def create_lists(self):
        file = open(self.filename, 'r')
        for line in file:
            if line[0] != '>':

                self.seq_list.append(line.strip())
        return self.seq_list

    def finish_work(self,who):
        print (len(self.seq_list))
        while len(self.seq_list) > 0:
            SL = self.seq_list.pop()
            data = Get_Seq_Data(self.mRNA, SL.seq, self.Feature, self.left, self.right)
            self.data_list.append(data)

    def start(self):
        self.create_lists()
        pool = mp.Pool(processes=self.cpucore)
        for i, member in enumerate(self.members):
            pool.apply_async(self.finish_work, (member,))
        pool.close()
        pool.join()


    #将数据返回去
    def GetData(self):
        pos_num = 0
        neg_num = 0
        for data in self.data_list:
            if data.label == 1:
                pos_num=pos_num+1
            elif data.label == 0:
                neg_num=neg_num+1
        print(self.filename + '\tpos_num:' + str(pos_num) + '\tneg_num:' + str(neg_num))
        return self.data_list
