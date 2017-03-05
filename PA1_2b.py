__author__ = 'khushboo_agarwal'

#import header files required
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os,sys
import PA1_1 as canny

'''
#since the gaussian function is normally distributed (Bell - curve), we get the random function to generate values in that particular shape
Also the standard deviation and mean as mentioned are 10 and 0 respectively. 
'''
def Gaussian_noise(S, I):
	mean = 0										
	standard_deviation = 10							#standard deviation is 10
	gauss = np.random.normal(mean, standard_deviation, S)  		
	gauss = np.reshape(gauss, S)
	gauss_image = gauss + I
	return gauss_image
'''
Implementing Salt and Pepper Noise
'''
def Salt_Pepper_Noise(S, I):
	probability = 0.10                              #since the probability given is 10%
	threshold = 1-probability
	s_p_image = np.zeros(S)
	for i in range(S[0]):
		for j in range(S[1]):
			rdn = np.random.random()
			if(rdn<probability):
				s_p_image[i,j] = 0
			elif(rdn>threshold):
				s_p_image[i,j] = 255
			else:
				s_p_image[i,j] = I[i,j]
	return s_p_image

def Quant_Anal(Output, C):
	O = np.array(Output)
	S= np.shape(O)
	for i in range(S[0]):
		for j in range(S[0]):
			if(O[i,j] > 0):
				O[i,j] = 1
			else:
				O[i,j] = 0

	TP = 0.0		#true positive
	TN = 0.0		#true negative
	FP = 0.0		#false positive
	FN = 0.0		#false negative

	for i in range(S[0]):
		for j in range(S[1]):
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
	print "F-score", F
	print "Mathew Correlation Coefficient", MCC

def Plotting_Results(I1, I2, G_1, G_2, s_p_image1, s_p_image2):
	plt.figure()   									# plotting all the images in subplot
	plt.subplot(2,3,1)
	plt.title('image 1')
	plt.imshow(I1,cmap = cm.gray)
	plt.subplot(2,3,2)
	plt.title('image 1 with Gaussian noise')
	plt.imshow(G_1,cmap = cm.gray)
	plt.subplot(2,3,3)
	plt.title('Image 1 with Salt and Pepper')
	plt.imshow(s_p_image1,cmap = cm.gray)
	plt.subplot(2,3,4)
	plt.title('image 2')
	plt.imshow(I2,cmap = cm.gray)
	plt.subplot(2,3,5)
	plt.title('image 2 with Gaussian noise')
	plt.imshow(G_2,cmap = cm.gray)
	plt.subplot(2,3,6)
	plt.title('Image 2 with Salt and Pepper')
	plt.imshow(s_p_image2,cmap = cm.gray)
	plt.show()


def noise_calc_quantative_analysis():
	Input_1 = Image.open("input_image.jpg").convert('L')
	Output_1 = Image.open("output_image.png").convert('L')
	I1 = np.array(Input_1)
	S_1 = np.shape(I1)
	G_1 = Gaussian_noise(S_1, I1)
	s_p_image1 = Salt_Pepper_Noise(S_1, I1)
	C11 = canny.Canny_Edge_(G_1, 1, 20, 30)
	C12 = canny.Canny_Edge_(s_p_image1, 1, 20, 30)
	print("The Quantative Analysis for Image 1 is as shown below:")
	print("For image1 with Gaussian noise:")
	Quant_Anal(Output_1, C11)
	print("For image2 with Salt and Pepper noise:")
	Quant_Anal(Output_1, C12)


	Input_2 = Image.open("input_image2.jpg").convert('L')
	Output_2 = Image.open("output_image2.png").convert('L')
	I2 = np.array(Input_2)
	S_2 = np.shape(I2)
	G_2 = Gaussian_noise(S_2, I2)
	s_p_image2 = Salt_Pepper_Noise(S_2, I2)
	C21 = canny.Canny_Edge_(G_2, 1, 20, 30)
	C22 = canny.Canny_Edge_(s_p_image2, 1, 20, 30)
	print("The quantitative analysis for the Image 2 is as shown below:")
	print("For image2 with Gaussian noise:")
	Quant_Anal(Output_2, C21)
	print("For image2 with Salt and Pepper Noise:")
	Quant_Anal(Output_2, C22)
	print("Plot for image and image with different noise")

	Plotting_Results(I1, I2, G_1, G_2, s_p_image1, s_p_image2)

if __name__ == "__main__":
	noise_calc_quantative_analysis()












