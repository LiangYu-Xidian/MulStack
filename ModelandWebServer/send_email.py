import sys
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from conf import *
import os
from threading import Thread
import socket
import time
import methods

# def scop_info(scop_file):
#     info = {}
#     last_name = ''
#     with open(scop_file) as fp:
#         for i in fp:
#             if i[0] == '>':
#                 i = i[1:].strip().split(' ', 2)
#                 last_name = i[0]
#                 info[last_name] = i[1:]
#             else:
#                 info[last_name].append(i.strip())
#     return info

# def extract_results(user_fold, cutoff=0.0):
#     scop_decription = scop_info('webserver/model/scop.fasta')
#     query_name = []
#     with open(os.path.join(user_fold, 'test.fasta'), 'r') as fp:
#         for i in fp:
#             if i[0] == '>':
#                 i = i[1:].strip().split()[0]
#                 query_name.append(i)
#     res = {}
#     with open(os.path.join(user_fold, 'LTR_input')) as seq_list, open(os.path.join(user_fold, 'score', 'LTR_score')) as seq_score:
#         for i, j in zip(seq_list, seq_score):
#             query = i.strip().split()[-5]
#             hit = i.strip().split()[-2]
#             score =float('%.2f' % float(j.strip().split()[2]))
#             if score< cutoff:
#                 continue
#             if query in res:
#                 res[query].append([hit, score])
#             else:
#                 res[query] = [[hit, score]]
#     res_html = []
#     for i in query_name:
#         hits = sorted(res[i], key=lambda d:d[1], reverse=True)
#         hits = [[j[0], scop_decription[j[0]][0], scop_decription[j[0]][1], scop_decription[j[0]][2], j[1], "images/"+j[0]+"_bio.jpg", "images/"+j[0]+"_asym.jpg"] for j in hits]
#         res_html.append([i, hits])
#     # query_name [hit_name scop_decriptrion scop_decriptrion scop_decriptrion score images images]
#     return res_html

def send_async_email(fromaddr, toaddr, subject, body, attachment, passwd):
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['subject'] = subject
    msg.attach(MIMEText(body, 'html'))
    msg.attach(attachment)
    server = smtplib.SMTP_SSL('smtp.163.com', '465')
    server.login('bionlpserver@163.com', passwd)
    server.sendmail(fromaddr, toaddr, msg.as_string())
    server.quit()

# def write_res(user_file,result):
#     with open(user_file,'w') as f:        
#         for i in result:
#             f.write('---------------------------------------------------------------------------------------\n')
#             f.write('The results of query protein ' + i[0] + ' is:')
#             f.write('\n---------------------------------------------------------------------------------------\n')
#             for j in i[1]:
#                 f.write('>' + j[0]+ ' ' + j[1] + ' ' + j[2] + '\n' + j[3]+ '\n')

def send_email(uip, cutoff, user_email):
    # user_dir = os.path.join(USER_FOLD, uip)
    # res = extract_results(user_dir, cutoff)
    download_path = os.path.join(WORK_FOLD, 'static/result/' + uip + '.txt')
    # write_res(download_path, res)
    if user_email != '':
        email_sended = 1
        fromaddr = 'bionlpserver@163.com'
        toaddr = user_email
        subject = 'Results from ProtDec-LTR'
        body = ''
        body +=    '<p>Dear user,<br /><br />'+ \
                   'Attached please find the results generated by ProtDec-LTR3.0 web-server. You can also access the results by clicking <a href="http://bioinformatics.hitsz.edu.cn/ProtDec-LTR3.0/result/'+ uip+ '?cutoff=' + str(cutoff)+ '"><font color="#123456"><b>Results</b><font></a><br /><br />'+ \
                   'Thank you for using ProtDec-LTR3.0 !<br /><br />'+ \
                   'With best regards,<br />'+ \
                   'Bioinformatics Group<br />'+ \
                   'School of Computer Science and Technology<br />'+ \
                   'Harbin Institute of Technology Shenzhen Graduate School</p>'
                 
        with open(download_path, 'r') as f:
            attachment = MIMEText(f.read())
            attachment.add_header('Content-Disposition', 'attachment', filename='result.txt')
            
        try:
            PASSWORD="bionlp2017"
            thread = Thread(target=send_async_email, args=[fromaddr, toaddr, subject, body, attachment, PASSWORD])
            thread.start()
        except socket.gaierror:
            pass
        except smtplib.SMTPServerDisconnected:
            pass
        except smtplib.SMTPException:
            pass

if __name__ == '__main__':
    user_email = sys.argv[1]
    cutoff = float(sys.argv[2])
    uip = sys.argv[3]
    user_dir = os.path.join(USER_FOLD, uip)
    res = methods.extract_results(user_dir, cutoff)
    download_path = os.path.join(WORK_FOLD, 'static/result/' + uip + '.txt')
    methods.write_res(download_path, res)
    # print user_email
    if user_email != 'no_email':
        send_email(uip, cutoff, user_email)