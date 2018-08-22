#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:35:47 2018

@author: zeke
"""

import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
def d_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x)) 

def xent(true,pred):
    return -((true*np.log(pred)) + ((1-true)*np.log(1-pred)))

lr = 0.1
iters = 1000
print_each = 100

w1 = np.random.rand(2,2)
w2 = np.random.rand(2,1)

for i in range(iters):
    #forward
    z1 = np.dot(X,w1)
    yh = sigmoid(z1)
    
    z2 = np.dot(yh,w2)
    y_ = sigmoid(z2)
    costo = np.mean(xent(Y,y_))
    acc = np.mean(Y==np.where(y_>=0.5,1,0))
    if i%print_each == 0:
        print("costo: {:.3f} \tAcc:{:.2f}".format(costo, acc))
    
    
    #backprop
    d2 = (Y-y_)*d_sigmoid(z2)
    dw2 = np.dot(yh.T,d2)
    d1 = np.dot(d2,w2.T) *d_sigmoid(z1)
    dw1 = np.dot(X.T, d1)
    
    #update
    w1 += lr*dw1
    w2 += lr*dw2
    
  
