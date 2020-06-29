import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from skimage.transform import resize

def min_max_check(img, value=0.1):
     for xl in range(np.shape(img)[1]):
          if np.sum(img[:,xl])>value:
               x_min = xl
               break
          
     for xl in range(np.shape(img)[1]):
          if np.sum(img[:,-xl])>value:
               x_max = np.shape(img)[1]-xl
               break

     for yl in range(np.shape(img)[0]):
          if np.sum(img[yl,:])>value:
               y_min = yl
               break

     for yl in range(np.shape(img)[0]):
          if np.sum(img[-yl,:])>value:
               y_max= np.shape(img)[0]-yl
               break
          
     return x_min, x_max, y_min, y_max

re_size = 64
gen_num = 100
for k in range(10,gen_num+1):
     if k%500==0: print(k)
     img = np.zeros((128,128,2),dtype=np.uint8)+255
     ref_img = np.zeros((128,128,2),dtype=np.uint8)+255

     ref_dimg = np.asarray(Image.open('dataset/dongdong/abnormal/abnormal_0/%d.PNG'%k,'r'))
     dimg = np.asarray(Image.open('dataset/dongdong/abnormal/abnormal_0/%d.PNG'%k,'r').convert('LA'))
     
     img[32:-32, 32:-32] = dimg
     img = -(img[:,:,0]-img[:,:,1])/255   ## make line : 1 / space : 0
     ref_img[32:-32, 32:-32] = ref_dimg
     
     xmin, xmax, ymin, ymax = min_max_check(img)
     croped_ref_img = ref_img[ymin-10:ymax+10,xmin-10:xmax+10]

##     plt.subplot(121)
##     plt.imshow(croped_ref_img, cmap='gray')

     croped_ref_img = Image.fromarray(croped_ref_img)
     croped_ref_img = croped_ref_img.resize((re_size,re_size), Image.ANTIALIAS)
##     print(np.shape(croped_ref_img))

##     plt.subplot(122)
##     plt.imshow(croped_ref_img, cmap='gray')
##     plt.show()

     croped_ref_img.save('dataset/dongdong2/abnormal/abnormal_0/%d.PNG'%k, cmap='gray')
     
