import os, shutil
import shlex
import subprocess
from conf import *
import pickle
import time
#from Bio import SeqIO
import threading
import numpy as np
import pandas as pd
import subprocess

from webserver.conf import MAX_NUM_OF_SEQUENCES, USER_FOLD, NLP_FOLD


def cmd(command):

    print(command)
    os.system('python /MulStack/webserver/run_predictor.py -uip ' + str(command))

def check_and_write(data, type_rna,path, max_num_of_sequences=MAX_NUM_OF_SEQUENCES):
    filename =  'human_'+ type_rna + '.fasta'
    with open(os.path.join(path, filename), 'w') as fp:
        line_num, sequence_num = 0, 0
        id_name = []
        first_is_name, need_line = False, False
        for i in data:
            line_num += 1
            i = i.strip()
            if len(i) == 0:
                continue
            if i[0] == '>':
                # check duplicted id
                if i in id_name:
                    return False, 'Duplicated ID: ' + i + '\nTwo sequences cannot share the same identifier!'
                else:
                    id_name.append(i)

                # check name
                if len(i.strip()) == 1:
                    return False, 'Line ' + str(line_num) + ' is wrong, it should be a name of protein sequence!'
                first_is_name = True
                sequence_num += 1

                # check sequence number
                if sequence_num > max_num_of_sequences:
                    return False, 'No more than ' + str(max_num_of_sequences) + ' protein sequences!'
                if sequence_num != 1:
                    fp.write('\n')
                i = i.replace('|', ' ')
                i = i.replace('/', ' ')
                i = i.replace('.', ' ')
                # fp.write(i + '\n')
                fp.write('>sequence_'+str(sequence_num)+ '\n')
            else:
                i = i.upper()
                if not first_is_name:
                    return False, 'Line 1 is wrong, it should be a name of rna sequence!'
                need_line = False
                fp.write(i)
        # for NLP word dataset by training data 
        fp.write('\n')
        with open(os.path.join(NLP_FOLD, filename), 'r') as fnlp:
            records = fnlp.read()
        fp.write(records)
    return True, None


def predict(uip):
    user_dir = USER_FOLD + '/' + uip
    th = threading.Thread(target=cmd, args=(user_dir + '/',))
    th.start()


def create_fold(path):
    subprocess.call('mkdir -m 777 ' + path, shell=True)


def getResult(jobid):
    user_dir = os.path.join(USER_FOLD, jobid)
    result_score = user_dir + "/result.csv"

    data = pd.read_csv(result_score, header=None).values
    headname = list(data[0])
    label = np.delete(data, 0, 0)

    return headname,label

if __name__=='__main__':
    uip = '127.0.0.1_miRNA_1672212166.21'
    predict(uip)


