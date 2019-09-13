import numpy as np
import os
import pandas
import matplotlib.pyplot as plt
import cv2
print('all libraries are import')


os.chdir('E:/DeepLearning/Open_CV')

#showing the main image.
image = cv2.imread('test_primary.png')
image=cv2.resize(image,(500,500))


image1 = cv2.imread('binary_img.jpg')
image1=cv2.resize(image1,(500,500))


cv2.imshow('main_image',image)
cv2.imshow('main_image of eye',image1)

#image shape.
karan=image.shape
print('shape of the image',karan)

#image size.
print('size of the image',image.size)

#image type.
print('type of image',image.dtype)

#Accessing and Modifying pixel values.
#command show what is the value of image pixel at[100,100].
pixel = image[100,100]
print('pixel value of [100,100]',pixel)

#modify the pixel values the same way.
image[253,253] = [255,255,255]
print('pixel value of [253,253]',image[253,253])

#Image Addition.
#for example image = img1+img2.
x = np.uint8([250])
y = np.uint8([10])

#normal addition will divide total divide by %256 because of uint8.
print('print_withcv2',cv2.add(x,y)) # 250+10 = 260 => 255
print('print_withoutcv2',x+y)          # 250+10 = 260 % 256 = 4


#Image Blending
blend = cv2.addWeighted(image,0.5,image1,0.5,0)
cv2.imshow('blending_outout',blend)
cv2.waitKey(0)
cv2.destroyAllWindows()

#bit wise operation
# I want to put logo on top-left corner, So I create a ROI
img1=image
image1=cv2.resize(image1,(500,500))
img2=image1
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 0, 200, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('output image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()






