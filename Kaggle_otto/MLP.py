import numpy as np
import theano
import theano.tensor as T

from Layer import Layer



class MLP(object):

  def __init__(self ,
  	           n_in ,
  	           n_out , 
  	           hidStruct ,
               hidAct ,
  	           input = None ,
               pDrop = 0.0):

    self.n_in  = n_in   # number of input dimensions 
    self.n_out = n_out  # number of output labels
    self.hidStruct = hidStruct # a list describe the nodes of each hidden layer
    self.hidAct = hidAct
    self.pDrop = pDrop

    self.struct =  [self.n_in] + self.hidStruct + [self.n_out] # nodes of all layers
    
    #####################################
    # construct multi-layer NNet
    #####################################
    self.layers = []
    for i in range( len(self.struct) - 2 ):
      self.layers.append(
      	                 Layer( 
      	                 	    name = ['layer ' , str(i+1)] ,
      	                      n_in  = self.struct[i]       ,
      	                      n_out = self.struct[i+1]     ,
                              pDrop = pDrop                ,
                              aFnt_No = self.hidAct[i]
      	                      )
                        )
    self.layers.append(
                       Layer(
                             name = 'output layer' ,
                             n_in = self.struct[-2] ,
                             n_out = self.struct[-1] ,
                             aFnt_No = -1        
                       	    )

                      )

    self.params = []
    for layer in self.layers:
      self.params += layer.params
    # print(self.params)


  def forwardProp(self , x):
    for layer in self.layers:
      x = layer.feed(x)
    return x

  def predict(self , x):
  	return self.forwardProp(x)

  def setDropoutOn(self):
    for i in range( len(self.struct) - 2 ):
      self.layers[i].setDropoutOn()
  
  def setDropoutOff(self):
    for i in range( len(self.struct) - 2 ):
      self.layers[i].setDropoutOff()


  def squareError(self , x , y):
    return T.mean((self.predict(x) - y)**2)
  
  def binary_crossentropy(self, x, y):
    return -(x*T.log(y)+(1.0-x)*T.log(1.0-y))

  def categorical_crossentropy(self, x, y):
    return T.sum(-T.log(x)*y , axis=1)
    # return T.log(x)[T.arange(y.shape[0]), y]

  def crossEntropyError(self , x , y):
    temp = self.predict(x)
    if self.n_out == 1 :
      return T.mean(self.binary_crossentropy(temp,y))
    else:
      return T.mean(self.categorical_crossentropy(temp,y))
      
  def getNumberOfHidden(self):
  	print('input:' , str(self.struct[0]))
  	print(self.struct[1:-2])
  	print('output:' , str(self.struct[-1]))
  	return self.struct

  def dAPreTraining(self, x , epoch = 1):
    for layer in self.layers:
      layer.doPreTraining(x,epoch )
      x = layer.getResult(x)


