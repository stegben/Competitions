import sys
from glob import glob
from os import walk, path
from random import random,shuffle
from time import time
import multiprocessing

from skimage import img_as_float
from skimage.io import imread,imshow
from skimage.exposure import equalize_adapthist, adjust_gamma, adjust_sigmoid
from skimage.morphology import erosion,dilation
from skimage.morphology import disk

from collections import Counter

import numpy as np

from sklearn.metrics import confusion_matrix

import theano

from keras.models import Sequential
from keras.optimizers import SGD	, Adadelta
from keras.layers.core import Dense, Activation, Flatten, Dropout , Merge
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D



train_folder_name = sys.argv[1]
output_model_name = sys.argv[3]
test_folder_name  = sys.argv[2]
predict_file_name = 'submit.csv'


epoch = 2000
n = 3000
proc_num = 6


"""
Create some helper function
1. returnProcessedImage
2. returnLabel
3. balanceClass
"""
def returnProcessedImage(que,folder,img_flist):
	X = []
	for fname in img_flist:
		cur_img = imread(folder+'/'+fname , as_grey=True)
		cur_img = 1 - cur_img

		######## randomly add samples

		# random add contrast
		r_for_eq = random()
		cur_img = equalize_adapthist(cur_img,ntiles_x=8,ntiles_y=8,clip_limit=(r_for_eq+0.5)/3)

		
		#random morphological operation
		r_for_mf_1 = random()
		if 0.05 < r_for_mf_1 < 0.25: # small vessel
			selem1 = disk(0.5+r_for_mf_1)
			cur_img = dilation(cur_img,selem1)
			cur_img = erosion(cur_img,selem1)
		elif 0.25 < r_for_mf_1 < 0.5: # large vessel
			selem2 = disk(2.5+r_for_mf_1*3)
			cur_img = dilation(cur_img,selem2)
			cur_img = erosion(cur_img,selem2)
		elif 0.5 < r_for_mf_1 < 0.75: # exudate
			selem1 = disk(9.21)
			selem2 = disk(7.21)
			dilated1 = dilation(cur_img, selem1)
			dilated2 = dilation(cur_img, selem2)
			cur_img = np.subtract(dilated1, dilated2)
		
		cur_img = img_as_float(cur_img)
		X.append([cur_img.tolist()])
	# X = np.array(X , dtype = theano.config.floatX)
	que.put(X)
	return X

def returnLabel(img_flist):
	Y = []
	for fname in img_flist:
		label = label_dict[fname.split('.')[0]]
		label_vec = [0]*3
		if label == 0 or label == 1:
			label_vec[0] = 1
		elif label == 2 or label == 3 :
			label_vec[1] = 1
		else:
			label_vec[2] = 1
		
		Y.append(label_vec)
	Y = np.array(Y , dtype = theano.config.floatX)
	return Y

def balanceClass(img_flist,label_dict):
	new_image_file_list = []
	for k in image_file_list:
		l = label_dict[k.split('.')[0]]
		if l == 2 or l==3:
			if random()<0.583:
				r = 5
			else:
				r = 4
		elif l==4 :
			r=40
		else:
			r=1
		for i in range(r):
			new_image_file_list.append(k)
	return new_image_file_list


"""
Start the whole training process
1. read in labels
2. get all image filename
3. balanced all class
"""
print('read in labels and balanced class.....')
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
image_file_list = balanceClass(image_file_list,label_dict)
# print(len(image_file_list))


"""
4. create CNN model
"""
print('create model...')
model = Sequential()

model.add(Convolution2D(8, 1, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(8, 8, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2), ignore_border=True))

model.add(Convolution2D(16, 8, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(16, 16, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(3, 3), ignore_border=True))

model.add(Convolution2D(32, 16, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(5, 5), ignore_border=True))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2048,2048,activation='tanh'))
model.add(Dropout(0.4))

model.add(Dense(2048,2048))
model.add(PReLU(2048))
model.add(Dropout(0.4))

model.add(Dense(2048,1024,activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(1024,3))
model.add(Activation('softmax'))

trainer = Adadelta(lr = 0.02 , rho = 0.97 , epsilon = 1e-8 )
# trainer = SGD(lr = 0.1, decay = 0.0 , momentum = 0.95 , nesterov = True)
model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)

if path.exists(output_model_name):
	model.load_weights(output_model_name)



print('start training by randomly read in ')

try:
	for i in range(epoch):
		print('real epoch: ',i)
		shuffle(image_file_list)

		for k in range(0,len(image_file_list),n):
			
			cur_flist = image_file_list[k:k+n]
			'''
			##### multiprocessing		
			X = []
			img_process_thread = []
			que_list = [multiprocessing.Queue() for a in range(proc_num)]
			l = len(cur_flist)/proc_num
			b = 0
			for h in range(0,len(cur_flist),l):
				p = multiprocessing.Process(target=returnProcessedImage,args=(que_list[b],folder,cur_flist[h:h+l]))
				img_process_thread.append(p)
				b+=1
			for p in img_process_thread:
				p.start()
			b = 0
			for p in img_process_thread:
				X += que_list[b].get()
				b+=1
				p.join()
			del que_list
			'''
			####### start training
			_ = multiprocessing.Queue()
			X = returnProcessedImage(_,folder,cur_flist)
			X = np.array(X,dtype=theano.config.floatX)
			Y = returnLabel(cur_flist)
			model.fit(X,Y,batch_size = 32,nb_epoch=1,shuffle=True,validation_split=0.0,show_accuracy=False)
			
			####### check training condition
			result = model.predict_classes(X,batch_size = 12,verbose=0)
			y_true = [a.index(1) for a in Y.tolist()]
			print(confusion_matrix(y_true,result.tolist()))

except KeyboardInterrupt:
	print('hey, Ctrl+C just been pressed, store the current model')

model.save_weights(output_model_name)

del label_dict
del image_file_list




"""
Start predicting...
"""
X_test = []
image_file_list = []
for (folder , _ , fnames) in walk(test_folder_name):
	image_file_list = fnames

f_sub = open('sub.csv','w')
f_sub.write('image,level\n')

for k in range(0,len(image_file_list),n):
	cur_flist = image_file_list[k:k+n]
	
	##### multiprocessing
	img_process_thread = []
	que_list = [multiprocessing.Queue()for a in range(proc_num)]
	l = len(cur_flist)/proc_num
	b = 0
	for h in range(0,len(cur_flist),l):
		p = multiprocessing.Process(target=returnProcessedImage,args=(que_list[b],folder,cur_flist[h:h+l]))
		img_process_thread.append(p)
		b+=1
	for p in img_process_thread:
		p.start()
	X = []
	b = 0
	for p in img_process_thread:
		X += que_list[b].get()
		b+=1
		p.join()

	####### start training
	X = np.array(X,dtype=theano.config.floatX)
	result = model.predict_classes(X,batch_size = 12,verbose=1).tolist()

	for fname,r in zip(cur_flist, result):
		f_sub.write(fname.split('.')[0]+','+str(r)+'\n')
	break

f_sub.close()
