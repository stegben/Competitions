import sys
from glob import glob
from os import walk
from random import random

from skimage.io import imread
from skimage.exposure import equalize_adapthist

from collections import Counter

import numpy as np

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


epoch = 10

############# get labels
print('read in labels.....')
f_lab = open('trainLabels.csv' , 'r')
f_lab.readline()
label_dict = {}
for line in f_lab:
	line = line.rstrip().split(',')
	label_dict[line[0]] = int(line[1])
f_lab.close()


############# get train photos
# image_file_list = glob(train_folder_name+'/*.jpeg')

image_file_list = []
for (folder , _ , fnames) in walk(train_folder_name):
	image_file_list = fnames

print('create model...')
model = Sequential()


model.add(Convolution2D(8, 3, 7, 7, border_mode='full')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(3, 3), ignore_border=True))
model.add(Dropout(0.25))

model.add(Convolution2D(16, 8, 11, 11, border_mode='full')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(5, 5), ignore_border=True))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 16, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(11, 11), ignore_border=True))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(288,1024,activation='relu'))
# model.add(PReLU(1024))
# model.add(Dropout(0.4))

model.add(Dense(1024,512,activation='relu'))
# model.add(PReLU(512))
# model.add(Dropout(0.4))

model.add(Dense(512,2))
model.add(Activation('softmax'))

trainer = Adadelta(lr = 0.05 , rho = 0.9 , epsilon = 1e-6 )
# trainer = SGD(lr = 0.1, decay = 0.0 , momentum = 0.95 , nesterov = True)
model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)


# model.load_weights(output_model_name)

# image_file_list = image_file_list[:4500]
print('start training by randomly read in ')
print('total images: ',len(image_file_list))
n = 1500
try:
	for i in range(epoch):
		print('real epoch: ',i)
		for k in range(0,len(image_file_list),n):
			X = []
			Y = []
			for fname in image_file_list[k:k+n]:
				label = label_dict[fname.split('.')[0]]
				"""
				if (label == 0) and (random()>0.02):
					continue
				if label == 1 and random()>0.1:
					continue
				if label == 2 and random()>0.05:
					continue
				if label == 3 and random()>0.5:
					continue
						 
				label_vec = [0]*5
				label_vec[label] = 1
				"""
				
				label_vec = [0]*2
				if label == 0 :
					label_vec[0] = 1
				else:
					label_vec[1] = 1
				

				cur_img = imread(folder+'/'+fname , as_grey=False)
				cur_img = equalize_adapthist(cur_img,ntiles_x=5,ntiles_y=5,clip_limit=0.1)
				cur_img = 1 - cur_img
				cur_img = np.ravel(cur_img,order='F')
				cur_img = np.reshape(cur_img,(3,512,512))
				X.append(cur_img.tolist())
				Y.append(label_vec)
			
			X_np = np.array(X , dtype = theano.config.floatX)
			del X
			Y_np = np.array(Y , dtype = theano.config.floatX)
			del Y
			
			model.fit(X_np,Y_np,batch_size = 13,nb_epoch=5,shuffle=True,validation_split=0.0,show_accuracy=True)
			result = model.predict_classes(X_np,batch_size = 12,verbose=0)
			print(Counter(result.tolist()))
			# model.fit(X,Y,batch_size = 4,nb_epoch=1,shuffle=True)
except KeyboardInterrupt:
	print('hey, Ctrl+C just been pressed')
	

# print(X)
####### release memory
del label_dict
del image_file_list


############ create model


print("start training...")


model.save_weights(output_model_name)


############ test now!
X_test = []
image_file_list = []
for (folder , _ , fnames) in walk(test_folder_name):
	image_file_list = fnames

f_sub = open('sub.csv','w')
f_sub.write([])
print('read in X, Y...')
for fname in image_file_list:
	cur_img = imread(folder+'/'+fname , as_grey=True)
	cur_img = 1 - cur_img
	X=cur_img.tolist()
	y = model.predict(X , batch_size=1)
	print(y)
# X_test_np = np.array(X_test , dtype = theano.config.floatX)

