import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
print('all libraries imported')

os.chdir('E:/DeepLearning/blindness/blood vessle')

#reading the image and resize the image.
img=cv2.imread('32_right.jpg',1)
img=cv2.resize(img,(600,600))


plt.subplot(1,5,1)
plt.imshow(img)
plt.title('main image')

#reading the image by adding 0 we can convert GRAY_SCALE and resize the image.

img2=cv2.imread('32_right.jpg',0)
img2=cv2.resize(img2,(600,600))

#split the variable into three main channel.
blue_image,red_image,green_image=cv2.split(img)


#printing the all channel image.
plt.subplot(1,5,2)
plt.imshow(blue_image)
plt.title('blue_image')

plt.subplot(1,5,3)
plt.imshow(red_image)
plt.title('red_image')

plt.subplot(1,5,4)
plt.imshow(green_image)
plt.title('green_image')

plt.subplot(1,5,5)
plt.imshow(img2)
plt.title('gray_image')

plt.show()





