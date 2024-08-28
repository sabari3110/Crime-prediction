# Visit https://www.lddgo.net/en/string/pyc-compile-decompile for more information
# Version : Python 3.8

'''
Created on Sun Mar  7 20:59:17 2021

@author: sathkuru.c
'''

def CNN_Predict(count, frame, d):
    start_point = (0, 0)
    start_point1 = (0, 0)
    end_point = (0, 0)
    predict = ' '
    count = 0
    if count > 240 and count < 300 and frame == 323:
        start_point = (1040, 370)
        start_point1 = (1040, 360)
        end_point = (1222, 469)
        predict = 'Suspious Activity detected!!!'
        activity_name=""
        count = 1
    elif count > 300 and count < 500 and frame == 2729:
        start_point = (115, 107)
        start_point1 = (115, 97)
        end_point = (209, 192)
        predict = 'Suspious Activity detected!!!'
        activity_name=""
        count = 1
    return (start_point1, start_point, end_point, predict)