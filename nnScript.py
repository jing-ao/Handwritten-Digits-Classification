import numpy as np
from numpy.core.numeric import ones
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))
    """

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    # Your code here.
    # if z>=0:
    return  1 / (1 + np.exp(-z))
    # # prevent exp overflow
    # else:
    #   return np.exp(z) / (1 + np.exp(z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    test_label = np.empty(len(mat['test0']))
    test_label.fill(0)
    test_data = mat['test0']

    for x in range(1, 10):
      keys = 'test' + str(x)
      temp = np.empty(len(mat[keys]))
      temp.fill(x)
      test_label = np.concatenate((test_label, temp), axis=None)
      test_data = np.vstack((test_data, mat[keys]))

    # every first 100 will be validation samples
    validation_label = np.empty(1000)
    validation_label.fill(0)
    validation_data = mat['train0'][:1000]

    for x in range(1, 10):
      keys = 'train' + str(x)
      temp = np.empty(1000)
      temp.fill(x)
      validation_label = np.concatenate((validation_label, temp), axis=None)
      validation_data = np.vstack((validation_data, mat[keys][:1000]))

    # rest rows will all serve as training samples
    train_label = np.empty(len(mat['train0']) - 1000)
    train_label.fill(0)
    train_data = mat['train0'][1000:]

    for x in range(1, 10):
      keys = 'train' + str(x)
      temp = np.empty(len(mat[keys]) - 1000)
      temp.fill(x)
      train_label = np.concatenate((train_label, temp), axis=None)
      train_data = np.vstack((train_data, mat[keys][1000:]))
    # Feature selection
    # Your code here.

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """
    % nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    """

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    training_data = np.c_[training_data, ones(len(training_data))]
    z = sigmoid(training_data @ np.transpose(w1))
    z = np.c_[z, ones(len(z))]
    output = sigmoid(z @ np.transpose(w2))

    n = len(training_label)
    y_label = np.zeros((n, n_class))
    for i in range(n):
      y_label[i][int(training_label[i])] = 1
    #obj_val = -1 * np.sum(y_label * np.log(output) + (1-y_label)*np.log(1-output)) / n
    obj_val = (-1/n)*np.sum(y_label * np.log(output) + (1-y_label)*np.log(1-output)) + (lambdaval/(2*n))*(np.sum(w1**2) + np.sum(w2**2))

    print('obj_val:',obj_val)

    theta = output - y_label
    grad_w2 = theta.T @ z
    grad_w2 = grad_w2 / n
    #print('grad_w2:', grad_w2.shape)

    z = np.delete(z, -1, axis=1)
    tmp_z = (1-z) * z
    tmp_w2 = np.delete(w2, -1, axis=1)
    tmp = theta @ tmp_w2
    tmp_p = (tmp_z * tmp)
    grad_w1 = tmp_p.T @ training_data

    grad_w1 = grad_w1 / n
    #print('grad_w1', grad_w1.shape)

    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """
    % nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels
    """

    labels = np.zeros(len(data))
    # Your code here
    data = np.c_[data, ones(len(data))]
    z = sigmoid(data @ np.transpose(w1))
    z = np.c_[z, ones(len(z))]
    output = sigmoid(z @ np.transpose(w2))
    labels = np.argmax(output, axis= 1)
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1] #784

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter


# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
# nnObjFunction(initialWeights, n_input, n_hidden, n_class, train_data, train_label, lambdaval)
opts = {'maxiter': 50}  # Preferred value.

for lambdaval in range(40,41):
  print('#################################################################################################################')
  print(f'##lambda = {lambdaval}#############################')
  print('#################################################################################################################')

  args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

  nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

  # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
  # and nnObjGradient. Check documentation for this function before you proceed.
  # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


  # Reshape nnParams from 1D vector into w1 and w2 matrices
  w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
  w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

  # Test the computed parameters

  predicted_label = nnPredict(w1, w2, train_data)

  # find the accuracy on Training Dataset

  print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

  predicted_label = nnPredict(w1, w2, validation_data)

  # find the accuracy on Validation Dataset

  print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

  predicted_label = nnPredict(w1, w2, test_data)

  # find the accuracy on Validation Dataset

  print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
selected_feature = np.arange(784)
parameter_list = [selected_feature, n_hidden, w1, w2, lambdaval]
pickle.dump(parameter_list, open("params.pickle", "wb"))
