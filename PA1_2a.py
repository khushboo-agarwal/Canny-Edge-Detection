'''
__author__ = 'khushboo_agarwal'
'''

from __future__ import division													#this will always get real values for division

#import header files required
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os,sys
import PA1_1 as canny


def Quant_Anal(Output, C):
	O = np.array(Output)
	S_1 = np.shape(O)
	for i in range(S_1[0]):
		for j in range(S_1[0]):
			if(O[i,j] > 0):
				O[i,j] = 1
			else:
				O[i,j] = 0

	TP = 0.0		#true positive
	TN = 0.0		#true negative
	FP = 0.0		#false positive
	FN = 0.0		#false negative

	for i in range(S_1[0]):
		for j in range(S_1[1]):
			if(O[i,j]>150 and C[i,j]>150):
				TP = TP+1
			if(O[i,j]<150 and C[i,j]<150):
				TN = TN+1
			if(O[i,j]>150 and C[i,j]<150):
				FN = FN+1
			if(O[i,j]<150 and C[i,j]>150):
				FP = FP+1
	TPR = TP/(TP+FN)
	TNR = TN/(TN+FP)
	PPV = TP/(TP+FP)
	NPV = TN/(TN+FN)
	FPR = FP/(FP+TN)
	FNR = FN/(FN+TP)
	FDR = FP/(FP+TP)

	#to calculate accuracy
	A = (TP+TN)/(TP+FN+TN+FP)
	#f-score calc
	F = 2*TP/(2*TP + FP +FN)
	MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

	print "Sensitivity", TPR
	print "Specificity", TNR
	print "Precision", PPV
	print "Negative Prediction Value", NPV
	print "Fall out", FPR
	print "false Negative Rate", FNR
	print "False discovery rate", FDR
	print "Accuracy", A
	print "F=score", F
	print "Mathew Correlation Coefficient", MCC


#import the test images given for this assignment
Input_1 = Image.open("input_image.jpg").convert('L')
C = canny.Canny_Edge_(Input_1, 1, 20, 30)

#converting the output image to binary image
Output_1 = Image.open("output_image.png")			#groundtruth image
Quant_Anal(Output_1, C)

Input_2 = Image.open("input_image2.jpg").convert('L')
C_2 = canny.Canny_Edge_(Input_1, 1, 30, 45)

Output_2 = Image.open("output_image2.png")
Quant_Anal(Output_2, C_2)
