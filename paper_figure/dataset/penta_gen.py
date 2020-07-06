import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image

np.random.seed(37)

def car2pol(x,y):
     r = np.sqrt(x**2 + y**2)
     theta = np.arctan2(y,x)
     return r, theta

def pol2car(r,theta):
     x = r*np.cos(theta)
     y = r*np.sin(theta)
     return x, y

def shift(r,theta,dx,dy):
     x_, y_ = pol2car(r,theta)
     x_ += dx
     y_ += dy

     r_, theta_ = car2pol(x_,y_)
     return r_, theta_

def variance(r,theta,order=0.2,rotation=False):
     dr = (2*np.random.rand(5)-1)*order
     dth = (2*np.random.rand(5)-1)*order/2

     r_ = r + dr
     th_ = theta + dth

     r_ = np.append(r_, r_[0])
     th_ = np.append(th_, th_[0])
     
     if rotation:
##          th_ += (np.pi/8)+(2*np.random.rand()-1)*order
         th_ += (2*np.random.rand()-1)*(order/1.5)
         
     return r_, th_

r_std = np.float32([2,2,2,2,2])
th_std = np.float32([0,2*np.pi/5,2*np.pi*2/5,2*np.pi*3/5,2*np.pi*4/5])+np.pi/2

re_size = 64
gen_num = 5000
for k in range(1,gen_num+1):
     if k%100==0: print(k)
     ax = plt.subplot(111, projection='polar')
     linewidth = np.random.rand()*0.8+0.2     ### trial7 setup

     r_1, th_1 = variance(r_std, th_std, order = 0.3, rotation=True)
     ax.plot(th_1, r_1,'k', linewidth=linewidth)

     r_2, th_2 = variance(r_std,th_std, order = 0.3, rotation=True)

     dx_var = (2*np.random.rand()-1)*0.25
     dy_var = (2*np.random.rand()-1)*0.3
     r_2, th_2 = shift(r_2, th_2, dx=2.7+dx_var,dy=dy_var)
     ax.plot(th_2, r_2,'k', linewidth=linewidth)

     ax.set_rmax(8)
     ax.grid(False)
     ax.set_rticks([10])
     
     plt.savefig('ideal_penta/ideal_%d.png'%k)
##     plt.show()
     plt.clf()
     
     img = Image.open('ideal_penta/ideal_%d.png'%k,'r')
     img = img.crop((270-np.random.randint(15), 180-np.random.randint(15),
                     445+np.random.randint(15), 300+np.random.randint(15)))
     img = img.resize((re_size,re_size), Image.ANTIALIAS)
     plt.imshow(img, cmap='gray')
     img.save('ideal_penta/ideal_%d.png'%k, cmap='gray')
     plt.clf()
