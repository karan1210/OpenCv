import cv2
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import try_all_threshold

path = E:\DeepLearning
os.chdir('E:\DeepLearning')

 
#from skimage import data

img = cv2.imread('binary_img.jpg')

#grayscale = rgb2gray(img)

input_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##skimage.filters.thresholding.threshold_isodata
##skimage.filters.thresholding.threshold_li
##skimage.filters.thresholding.threshold_mean
##skimage.filters.thresholding.threshold_minimum
##skimage.filters.thresholding.threshold_otsu
##skimage.filters.thresholding.threshold_triangle
##skimage.filters.thresholding.threshold_yen

#we are getting all this filters using the try_all_threshold

#spliting the data into fig and ax
fig, axis = try_all_threshold(input_img)
plt.show()

#save the image into directory.
cv2.imwrite(path+"all_figure_.jpg",fig)
