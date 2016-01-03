import sys
from glob import glob
from os import walk
from random import random

from skimage.io import imread

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
X = []
Y = []
image_file_list = []
for (folder , _ , fnames) in walk(train_folder_name):
	image_file_list = fnames

# image_file_list = image_file_list[:4500]
print('read in X, Y...')
for fname in image_file_list:
	label = label_dict[fname.split('.')[0]]
	if (label == 0) and (random()>0.03):
		continue
	if label == 1 and random()>0.2:
		continue
	if label == 2 and random()>0.09:
		continue
	cur_img = imread(folder+'/'+fname , as_grey=True)
	X.append([cur_img.tolist()])
	 
	"""
	label_vec = [0]*5
	label_vec[label] = 1
	"""
	label_vec = [0]*3
	if label == 0 or label == 1:
		label_vec[0] = 1
	elif label == 2 or label == 3 :
		label_vec[1] = 1
	else:
		label_vec[2] = 1
	
	Y.append(label_vec)
	

# print(X)
####### release memory
del label_dict
del image_file_list


############ create model
print('create model...')
model = Sequential()

left = Sequential()
right = Sequential()

left.add(Convolution2D(8, 1, 3, 3, border_mode='full')) 
left.add(Activation('relu'))
left.add(Convolution2D(8, 8, 3, 3)) 
left.add(Activation('relu'))
left.add(MaxPooling2D(poolsize=(16, 16), ignore_border=True))
left.add(Dropout(0.25))

right.add(Convolution2D(8, 1, 11, 11, border_mode='full')) 
right.add(Activation('relu'))
right.add(Convolution2D(8, 8, 5, 5)) 
right.add(Activation('relu'))
right.add(MaxPooling2D(poolsize=(32, 32), ignore_border=True))
right.add(Dropout(0.25))

model.add(Merge([left,right],mode='concat'))

model.add(Flatten())

model.add(Dense(16384,512))
model.add(PReLU(512))
model.add(Dropout(0.4))

model.add(Dense(512,256,activation = 'tanh'))
model.add(Dropout(0.4))

model.add(Dense(256,3))
model.add(Activation('softmax'))

trainer = Adadelta(lr = 5.0 , rho = 0.95 , epsilon = 1e-5 )
# trainer = SGD(lr = 0.1, decay = 0.0 , momentum = 0.95 , nesterov = True)
model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)


X_np = np.array(X , dtype = theano.config.floatX)
del X
Y_np = np.array(Y , dtype = theano.config.floatX)
del Y

print("start training...")
model.fit([X_np,X_np],Y_np,batch_size = 32,nb_epoch=10,shuffle=False,validation_split=0.2,show_accuracy=True)
# model.fit(X,Y,batch_size = 4,nb_epoch=1,shuffle=False)

model.save_weights(output_model_name)


############ test now!
X_test = []
image_file_list = []
for (folder , _ , fnames) in walk(test_folder_name):
	image_file_list = fnames

print('read in X, Y...')
for fname in image_file_list:
	cur_img = imread(folder+'/'+fname , as_grey=True)
	X_test.append([cur_img.tolist()])

X_test_np = np.array(X_test , dtype = theano.config.floatX)

ans = model.predict([X_test_np, X_test_np],batch_size=4)
print(ans)
