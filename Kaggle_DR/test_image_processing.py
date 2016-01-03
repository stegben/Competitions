import sys
from glob import glob
from os import walk
from random import random,shuffle

from skimage import img_as_float
from skimage.io import imread,imshow,imsave
from skimage.exposure import equalize_adapthist, adjust_gamma, adjust_sigmoid
from skimage.morphology import erosion,dilation
from skimage.morphology import disk

import numpy as np


train_folder_name = sys.argv[1]


############# get labels
print('read in labels.....')
f_lab = open('trainLabels.csv' , 'r')
f_lab.readline()
label_dict = {}
for line in f_lab:
    line = line.rstrip().split(',')
    label_dict[line[0]] = int(line[1])
f_lab.close()


image_file_list = []
for (folder , _ , fnames) in walk(train_folder_name):
	image_file_list = fnames



for fname in image_file_list:

	label = label_dict[fname.split('.')[0]]
	
	cur_img = imread(folder+'/'+fname , as_grey=True)
	
	cur_img = 1 - cur_img
	######## randomly add samples

	# random add contrast
	r_for_eq = random()
	cur_img = equalize_adapthist(cur_img,ntiles_x=8,ntiles_y=8,clip_limit=(r_for_eq+0.5)/3)
	cur_img = img_as_float(cur_img)
	# imsave(str(label)+'/'+fname,cur_img)

	
	#random morphological operation
	r_for_mf_1 = random()
	if r_for_mf_1 < 0.25: # small vessel
		selem1 = disk(0.25+r_for_mf_1/2)
		cur_img = erosion(cur_img,selem1)
		cur_img = dilation(cur_img,selem1)
		cur_img = img_as_float(cur_img)
		# imsave(str(label)+'/'+'lv/'+fname,cur_img)
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
	
	