# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 09:39:44 2021

@author: sathkuru.c
"""

import numpy as np
import cv2
import imutils 
import time
from datetime import datetime
current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
print("Current date & time : ", current_datetime)
 
# convert datetime obj to string
str_current_datetime = str(current_datetime)
file = open("time.txt", 'r')
#data = file.readline()
#if len(data) < 1 :
file = open("time.txt", 'a')
file.write("work")
file.writelines(str_current_datetime + '\n')
file.close()
