"""
2015 Spring MLDS HW1
Apr.9 2015
b00901146 Chu Po-Hsien
b00901104 Hwang Kado
b00901010 Lin Hsien-Chin

See parameter setting part, then you'll know what can modify

"""

import numpy as np
import itertools as itr
import theano
import theano.tensor as T
from time import time
import matplotlib.pyplot as plt
import csv
import os.path
import cPickle as pickle
import random

# function we done by ourselve
from Layer import Layer
from MLP import MLP
import MLPtrainer

######################################################################################################################
#
# parameter setting
#
######################################################################################################################

##### basic parameter
pepoch = 3 # epoch for pretraining
epoch = 400
batch_size = 100
hiddenStruct = [1500,1500,1000,1200]
hiddenActFunc = [0,0,0,0] 
'''
  0: sigmoid
  1: tanh
  2: relu
  3: maxout(not done yet)

  -1: softmax
'''
alpha = 0.3
momentum = 0.95
L1 = 0.0
L2 = 0.002
skip_rate = 0.9
dropout = 0.5
useState = True # use state label(1943) or not(48)
doPT = False # do pre training or not
doDrop = False # do neuron dropout
holdout = 0.1 # cross-validation ratio, 0~1


###### feature parameter
feature = 'fbank' 
pre = 6 # time-window pre frame number
post = 6 # time-window post frame number

###### programming parameter
seePred = False # show training accuracy while in training process. 
               # to get higher speed, set False
showProgress = 40000 # tell user how many instances has been trained


"""
TODOs

Regularization:
1. maxout
2. new normalization

Different Model
1. CNN
2. LSTM

feature preprocessing: 
1. HHT on raw wav, T-F plot
2. fMLLR
3. i-vector

pre-training:
1. DBM
2. in-process pre-training
3. Deep AutoEncoder
 
learning method:
1. heat transportation model
2. Ada Gradient descent
"""

##################################################################################################################
#
# Initialization
#
##################################################################################################################

###### stored file name setting
outfilename = 'result_' \
             + feature  \
             + str(pre+post+1) \
             + 'frames' \
             +('_withPT' if doPT else'') \
             + '.csv'
modelName = feature \
          + ('_stateLabel_' if useState else '_') \
          + str(pre+post+1) \
          + 'frames' 
for num in hiddenStruct:
  modelName = modelName + '_' + str(num)
modelName += '.pickle'

##### feature number setting
if feature == 'fbank':
  raw_feat_num = 69
elif feature == 'mfcc':
  raw_feat_num = 39
else :
  raise ValueError('Wrong feature file name')

##### normalization
if os.path.exists(('./' + feature + 'Norm.pickle')):
  fnorm = open(('./' + feature + 'Norm.pickle') , 'r')
  norm , minv = pickle.load(fnorm)
else: 
  ftr = open( ('./'+ feature +'/train.ark') , 'r') #training set
  d = [[] for x in range(raw_feat_num)]
  for row in ftr:
    row = row.rstrip().split(" ")
    feat = [ float(a) for a in row[1:] ]
    for h in range(raw_feat_num):
    	d[h].append(feat[h])
  norm = [1] * raw_feat_num
  minv = [0] * raw_feat_num
  for ind in range(raw_feat_num):
    norm[ind] = 4 / (max(d[ind]) - min(d[ind]))
    minv[ind] = min(d[ind])
  ftr.close()
  fnorm = open((feature + 'Norm.pickle') , 'w')
  pickle.dump((norm,minv) , fnorm)
fnorm.close()

def normalize(x):
  return [(a-m)*b-2 for a,b,m in zip(x,norm,minv)]

####### all files used
ftr = open(('./'+ feature +'/train.ark') , 'r') #training set
fte = open(('./'+ feature +'/test.ark')  , 'r') # testing set
if useState:
  flab = open('./state_label/train.lab'  , 'r') # label
else:
  flab = open('./label/train.lab'  , 'r') # label
fmap = open('./phones/state_48_39.map' , 'r') # label mapping 48-39

######## model initialization
model = MLP(
	        n_in=raw_feat_num*(pre+post+1) ,
	        n_out=1943 if useState else 48 ,
	        hidStruct = hiddenStruct       ,
          hidAct = hiddenActFunc         ,
          pDrop = dropout if doDrop else 0.0                
	       )

###### prediction function (theano type)
x = T.matrix('x')
pred = theano.function([x] , model.predict(x))

######## model trainer initialization 
trainer = MLPtrainer.MLPtrainer(
	       net = model ,
	       learning_rate = alpha ,
	       momentum = momentum ,
	       L1 = L1 , 
	       L2 = L2 )


# label initialization
labelSet  = [ 'aa','ae', 'ah','ao', 'aw','ax','ay', 'b',
	      'ch','cl',  'd','dh', 'dx','eh','el','en',
	     'epi','er', 'ey', 'f',  'g','hh','ih','ix',
	      'iy','jh',  'k', 'l',  'm','ng', 'n','ow',
              'oy', 'p',  'r','sh','sil', 's','th','t',
	      'uh','uw','vcl', 'v',  'w', 'y','zh', 'z']
if useState:
  labmap = {}
  for row in fmap:
    l = row.rstrip().split("\t")
    labmap[ l[0] ] = l[2]
else: 
  labmap = {}
  for row in fmap:
    l = row.rstrip().split("\t")
    labmap[ l[1] ] = l[2]
 
###### create label dictionary
label_dict = {}
for row in flab:
  lab = row.rstrip().split(",")
  label_dict[lab[0]] = lab[1]

flab.close()
fmap.close()

#create time-windowed frame
print("creating time-windowed frame...")
t1 = time()
feat_raw = []
name = []
for row in ftr :
  row = row.rstrip().split(" ")
  feat = normalize([ float(a) for a in row[1:] ])
  feat_raw.append( feat )
  name.append(row[0])
feat_new = []
r = len(feat_raw)
for i in range(r):
  new_feat = []
  for k in reversed(range(0,pre)):
    new_feat += feat_raw[i-k-1]
  new_feat += feat_raw[i]
  for h in range(0,post): 
    new_feat += feat_raw[(i+h+1)%r]
  feat_new.append(new_feat)
ftr.close()
t2 = time()
print('time-windowed frame creation spend: %0.2f s' % (t2-t1))

######## split training and CV
name_list = []
for n in name:
  tmp = n.split("_")
  if tmp[0] not in name_list:
    name_list.append(tmp[0])

cv_name_list = []
for nl in name_list:
  r = random.random()
  if r < holdout:
    cv_name_list.append(nl)

tr_name_list = list(set(cv_name_list)^set(name_list))

cv_name , cv_feat_new, tr_name , tr_feat_new = [] , [] , [] ,[]

for n,f in itr.izip(name,feat_new):
  if n.split("_")[0] in cv_name_list:
    cv_name.append(n)
    cv_feat_new.append(f)

for n,f in itr.izip(name,feat_new):
  if n.split("_")[0] in tr_name_list:
    tr_name.append(n)
    tr_feat_new.append(f)


##################################################################################################################
#
# Training
#
##################################################################################################################

i = 1
correct = 0
error = []
feat_batch = []
t11 = time()
##### start pre-training
print('start pre-training...')
while doPT and pepoch>=1 :
  print(".............epoch left : %d" % pepoch )
  # pre-training
  for row in tr_feat_new :
    # randomly skip instances to avoid overfitting
    r = random.random()
    if r < skip_rate:
      continue
    
    feat_batch.append(row)

    # batch full, train the model!
    if len(feat_batch) >= batch_size :
      X = np.array(feat_batch , dtype = theano.config.floatX )
      model.dAPreTraining(X)
      feat_batch = []
    
    # show progress
    if i % 10000 == 0:
      print('pre-train instances number: %d' % i)

    i += 1
	# end for loop: loop through all file
  pepoch -= 1
t22 = time()

if not doDrop:
  model.setDropoutOff()
  # redefine trainer since the argument has changed
  trainer = MLPtrainer.MLPtrainer(
         net = model ,
         learning_rate = alpha ,
         momentum = momentum ,
         L1 = L1 , 
         L2 = L2 )

i=1
feat_batch=[]
lab_batch = []
print('start Training by FBank conv...')
t1 = time()
for k in range(epoch):
  try: 
    print("................start epoch: %d" % (k+1) )
    # iteration through all training examples
    for n,row in itr.izip(tr_name,tr_feat_new) :
      # randomly skip instances to avoid overfitting
      r = random.random()
      if r < skip_rate:
        continue
      feat_batch.append( row )
      
      # label vector
      if useState:
        temp = [0] * 1943
        temp[ int(label_dict[n]) ] = 1
      else:
        temp = [0] * 48
        temp[ labelSet.index(label_dict[n]) ] = 1
      lab_batch.append(temp)

      # batch full, train the model!
      if len(lab_batch) >= batch_size :
        # get a predition on tr
        if seePred:
          p = pred(np.array([row],dtype = theano.config.floatX ))
          largestInd = max( (v, i) for i, v in enumerate(p[0]) )[1]
          if largestInd == (int(label_dict[n]) 
                            if useState else labelSet.index(label_dict[n])):
            correct += 1

        X = np.array(feat_batch , dtype = theano.config.floatX )
        Y = np.array(lab_batch , dtype = theano.config.floatX )

        e = trainer(X , Y)
        error.append(e)
  	  
        feat_batch = []
        lab_batch = []

      # show progress
      if i % showProgress == 0:
        print('train instances number: %d' % i)
        # print(model.layers[2].mask)
        print('error: %f' % sum(error[ (len(error)-100) :]))
        if seePred:
          acc = float(correct) / (showProgress/batch_size)          
          print('accuracy: '+str(acc))
          correct = 0
      i += 1
  	# end for loop: loop through all file
    if (k+1) % 5 == 0:  
      if doDrop:
        model.setDropoutOff()
        # redefine trainer since the argument has changed

      cv_correct = 0
      cv_instance = len(cv_name)
      for cvn,cvrow in itr.izip(cv_name,cv_feat_new) :
        # print(cvn)
        p = pred(np.array([cvrow],dtype = theano.config.floatX ))
        largestInd = max( (v, i) for i, v in enumerate(p[0]) )[1]
        if largestInd == (int(label_dict[cvn]) 
                          if useState else labelSet.index(label_dict[cvn])):
          cv_correct += 1
      cvacc = float(cv_correct) / cv_instance
      if doDrop:
        model.setDropoutOn()
      print('.........................')
      print('CV accuracy:' + str(cvacc))
      print('.........................')
  # manually interupt the model learning
  except KeyboardInterrupt:
    temp_epoch = k
    print("Ctrl+C detect")
    break
  # cross validation 
  

# end for loop: epoch  
t2 = time()
train_time = t2-t1
print('training time spend: %0.2f s' % train_time)
pre_train_time = t22-t11
print('pre training time spend: %0.2f s' % pre_train_time)


##################################################################################################################
#
# Testing
#
##################################################################################################################

model.setDropoutOff()
###### write in file preparation
fresult = open(outfilename , 'w')
w = csv.writer(fresult)
w.writerow(['Id' , 'Prediction'])

feat_test_raw = []
name = []
for row in fte :
  row = row.rstrip().split(" ")
  feat = normalize([ float(a) for a in row[1:] ])
  feat_test_raw.append( feat )
  name.append(row[0])
feat_test_new = []
r = len(feat_test_raw)
for i in range(r):
  new_feat = []
  for k in reversed(range(0,pre)):
    new_feat += feat_test_raw[i-k-1]
  new_feat += feat_test_raw[i]
  for h in range(0,post): 
    new_feat += feat_test_raw[(i+h+1)%r]
  feat_test_new.append(new_feat)
fte.close()

if doDrop:
  model.setDropoutOff()

###### start predicting 
i = 0
print('start predicting...')
for n,row in itr.izip(name,feat_test_new):
  
  # feature vector
  input = np.array([row] , dtype = theano.config.floatX)
  
  # get prediction distribution
  out = pred(input)
  largestInd = max( (v, i) for i, v in enumerate(out[0]) )[1]
  if useState:
    label = labmap[ str(largestInd) ]
  else:
    label = labmap[ labelSet[largestInd] ]
  
  result = []
  result.append(n) # id
  result.append(label) # label
  
  w.writerow(result)

  if i % showProgress == 0:
    print('test instances number: %d' % i)
  i += 1

print('predicting done.')
netFile = open(modelName , 'w')
pickle.dump(model,netFile)
fresult.close()
plt.plot(error)
plt.show()
