#!/usr/bin/env python
"""Python wrapper to convert stereo to depth map
Ref: http://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html#gsc.tab=0
"""
import cv2
from matplotlib import pyplot as plt


imgL = cv2.imread('./example_data/example_data/Drill_Left_10_n1p190_d0p784.png')
imgR = cv2.imread('./example_data/example_data/Drill_Right_10_n1p190_d0p784.png')

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)


imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

