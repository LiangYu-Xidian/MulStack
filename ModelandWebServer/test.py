
import os, shutil
import shlex
import subprocess
from conf import *
import pickle
import time
from Bio import SeqIO
import tmp2
import numpy as np

def extract_results(user_dir, uip):
    result = []
    seq = []
    result_dir = user_dir + '/result.txt'
    input_file = user_dir + '/test.fasta'
    with open(input_file) as f:
        for line in f:
            if line[0].isalpha():
                seq.append(line)
    num = -1
    with open(result_dir) as f:
        for line in f:
            if line[0] == '>':
                num += 1
                temp = []
                name = line.strip('>').strip('\n').strip('\r')
            if line[0].isdigit():
                result_list = line.strip('\n').strip(' ').split(' ')
                temp.append(name)
                temp.append(seq[num])
                temp.append(str(result_list))
                cout_np = []
                for i in range(len(result_list)):
                    cout_np.append(i)
#                print cout_np
                print list(seq[num])
                print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
                for i in range(len(list(seq[num]))):
                    if i%10 == 0:
                        print ''.join(list(seq[num])[i:i+10])
                print '\n\n'
                result.append(temp)
#    shutil.move(user_dir, "webserver/static/result")

#    os.mkdir("webserver/static/result/" + uip + "/sin_result")
#    for i in range(len(result)):
#        sin_res = open("webserver/static/result/" + uip + "/sin_result/" + str(i), 'w')
#        sin_res.write('>' + result[i][0] + '\n')
#        sin_res.write(result[i][1] + '\n')
#        for j in range(len(result[i][2])):
#            sin_res.write(result[i][2][j] + ' ')
#        sin_res.write('\n')
    return result


def create_fold(path):
    cmd_args = shlex.split('mkdir -m 777 ' + path)
    subprocess.Popen(cmd_args).wait()

if __name__=='__main__':

#	Print f_download
#	F1 = open(f_download, 'w')
#	For i in range(len(pre_label)):
#		f1.write('>' + name_list[i] + '\n')
#		for j in range(len(pre_label[i])):
#				f1.write(str(pre_label[i][j]) + ' ')
#				f1.write('\n')
    uip = '10.249.43.140_1577625309.02'
    F_download = os.path.join('/var/www/RFPR-IDP/webserver/static/result/', uip + '_result_new.txt')
    aa = [1,2,3,4,5]
    np.savetxt(F_download, aa, fmt='%s')

#    #uip = '10.249.47.148_1577602221.67'
#    # predict(uip)
#    u = "/var/www/RFPR-IDP/webserver/static/result/10.249.47.148_1577004197.96/"
#    result = extract_results(u, '10.249.47.148_1577600890.94')
#    print result
  #  tmp2.result(uip)    

