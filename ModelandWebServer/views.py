from email import header
from pickle import NONE
from unittest import result
from flask import render_template,request, redirect, url_for, jsonify,send_from_directory, send_file
from matplotlib.pyplot import get
from webserver import app, methods
from conf import *
import os
import time
import smtplib
from threading import Thread
import socket
import subprocess
import shutil

from webserver.conf import USER_FOLD


@app.route('/')
@app.route('/home')
@app.route('/home/') 
def home():

    return render_template('home.html') 

@app.route('/myserver',methods=['GET', 'POST'])
def myserver():

    if request.method == 'POST':
        try :
            type_rna ='mRNA'

            user_ip_time = request.access_route[0]  +'_' + type_rna + '_' + str(time.time())
            user_dir = USER_FOLD + "/" + user_ip_time

            receive_seq = request.form['sequence']
            if receive_seq:
                sequences = receive_seq.replace('\r\n', '\n').split('\n')
            else:
                sequences = request.files['file']
            os.makedirs(user_dir,0o777)
            # create new temporary user floder
            check_result = methods.check_and_write(sequences, type_rna,user_dir)

            methods.predict(user_ip_time)
            print(user_ip_time)
            # run model and predicted
            return redirect(url_for('myresult', jobid=user_ip_time))

        except:
            return render_template('myserver.html')
    elif request.method == 'GET':
        return render_template('myserver.html')

@app.route('/document')
def document():
    return render_template('document.html')

@app.route('/contact')
@app.route('/contact/')
def contact():
    return render_template('contact.html')


@app.route('/about')
@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/myresult/')
@app.route('/myresult/<user_id>/download',methods=['GET'])
def user_result(user_id):
    user_dir = os.path.join(USER_FOLD, user_id)
    if not os.path.exists(user_dir):
        return render_template('Ooops.html')
    return send_file( user_dir + "/result.csv",mimetype='text/csv', attachment_filename='result.txt',as_attachment=True)


@app.route('/myresult/<jobid>',methods=['POST','GET'])
def myresult(jobid):

    user_dir = os.path.join(USER_FOLD, jobid)


    #user_dir = "/home/www/ncRNALocate-EL/webserver/temp/user\127.0.0.1_miRNA_1672197361.05"
    if not os.path.exists(user_dir):
        return render_template('Ooops.html')

    #request.method Get

    if request.method == 'POST':


        if os.path.exists(user_dir):
            result_score = user_dir + "/result.csv"
            if os.path.isfile(result_score):
                # read file
                header,data =methods.getResult(jobid)
                # return render_template('myresult.html',msger=msger)
                return render_template('myresult.html',data=data,header=header)
            else:
                return render_template('processing.html',jobid=jobid)
        else:
            return render_template('home.html')

    elif request.method == 'GET':
        # print(user_dir)
        result_score = user_dir + "/result.csv"


        if os.path.isfile(result_score) :
            # read file
            header, data =methods.getResult(jobid)
            # return render_template('myresult.html',msger=msger)
            return render_template('myresult.html',data=data,header=header)
        else:
            return render_template('processing.html',jobid=jobid)
