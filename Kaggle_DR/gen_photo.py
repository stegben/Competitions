import numpy as np
import sys
from glob import glob
from os import walk
from random import random,shuffle
from skimage import img_as_float
from skimage.io import imread,imshow,imsave
from skimage.exposure import equalize_adapthist, adjust_gamma, adjust_sigmoid
from skimage.morphology import erosion,dilation
from skimage.morphology import disk



f_img = open('./proc_small/green/sample/16_right.jpeg','r')
cur_img = imread('./proc_small/green/sample/16_right.jpeg' , as_grey=True)
cur_img = 1.0-cur_img
r_for_eq = random()
cur_img = equalize_adapthist(cur_img,ntiles_x=8,ntiles_y=8,clip_limit=0.01)
cur_img = img_as_float(cur_img)
# imsave('proc_small/green/'+str(r_for_eq)+'16_right.jpeg',cur_img)


#random morphological operation
r_for_mf_1 = random()
if r_for_mf_1 < 1.0: # small vessel
	selem1 = disk(2)
	cur_img_new = erosion(cur_img,selem1)
	cur_img_new = dilation(cur_img_new,selem1)
	

	cur_img_new = img_as_float(cur_img_new)
	cur_img = img_as_float(cur_img)
	imsave('proc_small/green/'+str(r_for_mf_1)+'sv.jpeg',cur_img-cur_img_new)

elif 0.25 < r_for_mf_1 < 0.5: # large vessel
	selem2 = disk(1.25+r_for_mf_1*1.5)
	cur_img = erosion(cur_img,selem2)
	cur_img = dilation(cur_img,selem2)
	cur_img = img_as_float(cur_img)
	# imsave(str(label)+'/'+'sv/'+fname,cur_img)
elif 0.5 < r_for_mf_1 < 0.75: # exudate
	selem1 = disk(4.605)
	selem2 = disk(3.604)
	# dilated1 = dilation(cur_img, selem1)
	# dilated2 = dilation(cur_img, selem2)
	dilated1 = erosion(cur_img, selem1)
	dilated2 = erosion(cur_img, selem2)
	cur_img = np.subtract(dilated1, dilated2)
	cur_img = img_as_float(cur_img)
	# imsave(str(label)+'/'+'ex/'+fname,cur_img)