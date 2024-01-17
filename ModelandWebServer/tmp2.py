

import methods
import os
import time


def result(uip):
    USER_FOLD = '/var/www/RFPR-IDP/webserver/temp/user/'
    user_dir = os.path.join(USER_FOLD, uip)
#    if request.method == 'POST':
 #       return render_template('home.html')
  #  elif request.method == 'GET':
    if 1:
        result_score = user_dir + '/result.txt'
        query_name = '3CSZA'
        print query_name
        print result_score
        if os.path.isfile(result_score):
            res = methods.extract_results(user_dir, uip)
            print res
            query_names = [i[0] for i in res]
            if query_name:
                for i in range(len(res)):
                    if res[i][0] == query_name:
                        res = res[i]
                        break
            else:
                res = res[0]
            print res
#            return render_template('result.html', result=res, uip=uip, query_names=query_names)
            #return render_template('result.html', result=res, uip=uip)
    #    else:
  #          return render_template('processing.html', uip=uip)
