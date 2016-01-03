import sys
from glob import glob
from os import walk
from random import random,shuffle

from skimage.io import imread
from skimage.exposure import equalize_adapthist, adjust_gamma, adjust_sigmoid

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


epoch = 2000
n = 1300

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

model.add(Dense(1024,5))
model.add(Activation('softmax'))

trainer = Adadelta(lr = 0.025 , rho = 0.97 , epsilon = 1e-8 )
# trainer = SGD(lr = 0.1, decay = 0.0 , momentum = 0.95 , nesterov = True)
model.compile(loss = 'categorical_crossentropy' , optimizer = trainer)

# model.load_weights(output_model_name)



# image_file_list = image_file_list[:4500]
print('start training by randomly read in ')

new_image_file_list = []
for k in image_file_list:
	l = label_dict[k.split('.')[0]]
	if l == 0:
		if random()<0.96:
			r = 1
		else:
			r = 0
	elif l == 1:
		r = 10
	elif l == 2:
		if random()<0.6:
			r = 5
		else:
			r = 4
	elif l ==3 :
		r = 28
	else:
		r = 34
	for i in range(r):
		new_image_file_list.append(k)



print(len(new_image_file_list))
image_file_list = new_image_file_list
del new_image_file_list
# print(Counter([label_dict[i.split('.')[0]] for i in image_file_list]))
try:
	for i in range(epoch):
		print('real epoch: ',i)
		shuffle(image_file_list)
		for k in range(0,len(image_file_list),n):
			X = []
			Y = []
			for fname in image_file_list[k:k+n]:
				label = label_dict[fname.split('.')[0]]
				
				cur_img = imread(folder+'/'+fname , as_grey=True)
				cur_img = 1 - cur_img

				# randomly add samples
				r_for_eq = random()

				if r_for_eq<0.3:
					cur_img = equalize_adapthist(cur_img,ntiles_x=5,ntiles_y=5,clip_limit=0.1)
				if 0.3<r_for_eq<0.4:
					cur_img = adjust_sigmoid(cur_img,cutoff=0.5, gain=10, inv=False)
				if 0.5<r_for_eq<0.6:
					cur_img = adjust_gamma(cur_img,gamma=0.5, gain=1)
				
				X.append([cur_img.tolist()])
				 
				
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
				"""
				Y.append(label_vec)
			
			X_np = np.array(X , dtype = theano.config.floatX)
			del X
			Y_np = np.array(Y , dtype = theano.config.floatX)
			del Y
			
			model.fit(X_np,Y_np,batch_size = 32,nb_epoch=3,shuffle=False,validation_split=0.0,show_accuracy=True)
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







print("start predicting...")


model.save_weights(output_model_name)


############ test now!
X_test = []
image_file_list = []
for (folder , _ , fnames) in walk(test_folder_name):
	image_file_list = fnames

f_sub = open('sub.csv','w')
f_sub.write('image,level\n')
print('read in X, Y...')
for fname in image_file_list:
	cur_img = imread(folder+'/'+fname , as_grey=True)
	cur_img = 1 - cur_img
	X=cur_img.tolist()
	y = model.predict_classes([X] , batch_size=1,verbose=0)
	w = fname.split('.')[0]+','+str(y[0])+'\n'
	# print(y)
# X_test_np = np.array(X_test , dtype = theano.config.floatX)

