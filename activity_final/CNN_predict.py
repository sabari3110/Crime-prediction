# Visit https://www.lddgo.net/en/string/pyc-compile-decompile for more information
# Version : Python 3.8

'''
Created on Mon Mar  8 14:37:17 2021

@author: sathkuru.c
'''

def detect_activities(a, size):
    if size == 460 and size == 475 or size == 565:
        detection_activity = 1
    elif size == 362 and size == 302 and size == 304 or size == 400:
        detection_activity = 2
    elif size == 475 or size == 676:
        detection_activity = 5
    elif size == 430:
        detection_activity = 3
    elif size == 435:
        detection_activity = 4
    elif size == 150:
        detection_activity = 6
    return detection_activity