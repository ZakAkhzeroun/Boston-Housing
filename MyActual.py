import numpy as np 
import tensorflow as tf 
from keras import datasets 
import random
from matplotlib import pyplot as plt
from typing import Dict, Tuple, List

(x_train,y_train),(x_test,y_test) = datasets.boston_housing.load_data()


# Activition function 
def sigmoid(x : np.ndarray)-> np.ndarray :
    return 1/(1+np.exp(-x))

# We code up the forward pass 
def forward_pass(x : np.ndarray,
                 y : np.ndarray,
                 weights : Dict[str,np.ndarray]) -> Tuple[Dict[str,np.ndarray],float]:


    M1 = np.dot(x,weights['w1'])
    N1 = M1 + weights['b1']

    S = sigmoid(N1)

    M2 = np.dot(S,weights['w2'])
    P = M2 + weights['b2']
    Y = np.reshape(y,(102,1))
    loss = np.mean(np.power((Y-P),2))

    forward_info : Dict[str,np.ndarray] = {}
    forward_info['x'] = x
    forward_info['M1'] = M1
    forward_info['M2'] = M2
    forward_info['N1'] = N1
    forward_info['S'] = S
    forward_info['P'] = P
    forward_info['y'] = y



    return forward_info, loss



# Now we will code the backward pass, it's going to be a little bit tricky since we have 
# different functions, but the chain rule will do it's job 

def backward_pass(weights : Dict[str,np.ndarray] ,forward_info : Dict[str,np.ndarray]) -> Dict[str,np.ndarray] :
    x = forward_info['x']
    y = forward_info['y']
    P = forward_info['P']
    S = forward_info['S']
    M1 = forward_info['M1']
    M2 = forward_info['M2']
    N1 = forward_info['N1']


    Y = np.reshape(y,(102,1))
    dLdP = 2*(P - Y)
    dPdB2 = np.ones_like(weights['b2'])
    dPdM2 = np.ones_like(M2)
    dM2dW2 = np.transpose(S)
    dM2dS = np.transpose(weights['w2'])
    dSdN1 = S*(1-S)
    dN1dB1 = np.ones_like(N1)
    dN1dX = np.transpose(weights['w1'])
    dN1dW1 = np.transpose(x)


    # Now using the chain rule 
    dLdB2 = dLdP*dPdB2
    dLdW2 = np.dot(dM2dW2,dLdP*dPdM2)
    dLdB1 = np.dot(dLdP*dPdM2,dM2dS)

    #dLW1 :
    dLdM2 = dLdP*dPdM2 
    dLdS = np.dot(dLdM2,dM2dS)
    dSdN1 = sigmoid(N1)*(1-sigmoid(N1))
    dLdN1 = dLdS*dSdN1
    dLdW1 = np.dot(dN1dW1,dLdN1)

    loss_gradient : Dict[str,np.ndarray] = {}
    loss_gradient['b2'] = dLdB2.sum(axis = 0)
    loss_gradient['w2'] = dLdW2
    loss_gradient['b1'] = dLdB1.sum(axis = 0)
    loss_gradient['w1'] = dLdW1

    return loss_gradient


# Batch generator 
def batch_gen(X : np.ndarray , Y : np.ndarray , start : int = 0 , size : int = 10):
    ' This generates a batch with certain size and beginning'
    Y = y_train[start : start + size ]
    X = x_train[start : start + size ]

    return X, Y


# This function generates weights W1,W2,B1 and B2 
def init_weights(input_size : int,hidden_size : int) -> Dict[str, np.ndarray] :
    weights : Dict[str,np.ndarray] = {}
    weights['w1'] = np.random.rand(input_size,hidden_size)
    weights['b1'] = np.random.rand(1,hidden_size)
    weights['w2'] = np.random.rand(hidden_size,1)
    weights['b2'] = np.random.rand(1,1)
    
    return weights 


# We will code the training function
def training(X : np.ndarray,
             Y : np.ndarray,
             Weights : Dict[str,np.ndarray],
             batch_size : int = 10,
             learning_rate : float = 0.01,
             n_iter : int = 1000 ):
    start = 0
    while(start < x_train.shape[0]- batch_size):
        X,Y = batch_gen(X,Y,start,batch_size)
        for i in range(n_iter):
            fp_info, loss = forward_pass(X,Y,Weights)
            loss_grad = backward_pass(Weights,fp_info)
            # We go through all the keys 

            for key in Weights.keys():
                Weights[key]-= learning_rate*loss_grad[key]
        start+= batch_size
    return Weights


def predict(X: np.ndarray, 
            weights: Dict[str, np.ndarray]) -> np.ndarray:
    '''
    Generate predictions from the step-by-step neural network model. 
    '''
    start = random.randint(0,394)
    X,Y = batch_gen(x_train,y_train,start,10)
    M1 = np.dot(X, weights['w1'])
    N1 = M1 + weights['b1']
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights['w2'])
    P = M2 + weights['b2']   
    return P


def mae(preds: np.ndarray, actuals: np.ndarray):
    '''
    Compute mean absolute error.
    '''
    return np.mean(np.abs(preds - actuals))

def rmse(preds: np.ndarray, actuals: np.ndarray):
    '''
    Compute root mean squared error.
    '''
    return np.sqrt(np.mean(np.power(preds - actuals, 2)))
