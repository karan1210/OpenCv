import cv2
import numpy
import os
import matplotlib.pyplot as plt
print('all libraries imported')

#changing the directory path.
os.chdir('E:\DeepLearning')
print(os.listdir('E:\DeepLearning'))

#read the image
img=cv2.imread('test.jpg',cv2.IMREAD_GRAYSCALE)
#cv2.imshow('gray_level',img)

#gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
axis,binary=cv2.threshold(img,127,256,cv2.THRESH_BINARY)
#cv2.imshow('', binary)

plt.subplot(1,2,1)
plt.imshow(img)
plt.title('gray image')

plt.subplot(1,2,2)
plt.imshow(binary)
plt.title('binary image')
plt.show()
#cv2.waitkey(0)
