import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
os.chdir('E:/DeepLearning/blindness/blood vessle')
img=cv2.imread('32_right.jpg',1)
img=cv2.resize(img,(600,600))
img2=cv2.imread('32_right.jpg',0)
img2=cv2.resize(img2,(600,600))


plt.subplot(1,5,1)
plt.imshow(img)
plt.title('main image')

blue,red,green=cv2.split(img)


plt.subplot(1,5,2)
plt.imshow(red)
plt.title('with only red parameter')

##k= img[:,:,:] = 0
##cv2.imshow('',img)
##cv2.imshow('',b)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
contrast_enhanced_red_fundus = clahe.apply(red)
contrast_enhanced_blue_fundus = clahe.apply(blue)
contrast_enhanced_green_fundus = clahe.apply(green)


                                          ########### for plotting the data ##############
plt.subplot(1,5,3)
plt.imshow(contrast_enhanced_red_fundus)
plt.title('red_fundus')

plt.subplot(1,5,4)
plt.imshow(contrast_enhanced_blue_fundus)
plt.title('blue_fundus')


plt.subplot(1,5,5)
plt.imshow(contrast_enhanced_green_fundus)
plt.title('green_fundus')

plt.show()

r1 = cv2.morphologyEx(contrast_enhanced_red_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 3)
R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 3)
r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 3)
R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 3)
r4 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20)), iterations = 1)

#cv2.imshow('LAST_CLOSE_MORPHOLOGY',R3)
f4 = cv2.subtract(r4,contrast_enhanced_red_fundus)
f5 = clahe.apply(f4)
#cv2.imshow('LAST_CLOSE_MORPHOLOGY',f5)

# Otsu's thresholding after Gaussian filtering

#all doing for thresholding.............

blur = cv2.GaussianBlur(f5,(1,1),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('gaussian',th3)

th = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('gaussian',th)

ret,f6 = cv2.threshold(f5,25,255,0)
#cv2.THRESH_BINARY = 0
cv2.imshow('',f6)
contours, hierarchy = cv2.findContours(f6,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print('Number of contours = ' + str(len(contours))  )







