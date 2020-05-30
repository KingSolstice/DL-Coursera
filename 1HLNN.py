# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


#Loads extra datasets for testing with the neural network
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "blobs"

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

if dataset == "blobs":
    Y = Y%2

#X, Y = load_planar_dataset()

#plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral);

#shape_X = X.shape
#shape_Y = Y.shape
#m = X.shape[1]  # training set size

########################################
#  LOGISTIC REGRESSION FAILURE EXAMPLE #
########################################
#clf = sklearn.linear_model.LogisticRegressionCV();
#clf.fit(X.T, Y.T);

#plot_decision_boundary(lambda x: clf.predict(x), X, Y[0])
#plt.title("Logistic Regression")

#//Print accuracy
#LR_predictions = clf.predict(X.T)
#print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#       '% ' + "(percentage of correctly labelled datapoints)")

#plt.show()


#Finds the sizes for each layer of the neural network (input neurons, hidden neurons, ouput neurons)
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

#Initializes the parameters for the NN
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    #W1 and W2 are randomly initialized so that they arent computing the same values during propagation
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

#Forward Propagation Method
def forward_propagation(X,parameters):

    #Parameter variable extraction
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #Propagation takes place
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1) #Hyperbolic tangent used for hidden layers for easier centering around zero
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2) #Sigmoid used for output layer in binary classification

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):

    m = Y.shape[1]

    #Calculation of cost via the loss function J = -(1/m) * sum(Y*Log(A2)+(1-Y)*Log(1-A2))
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
    cost = -(1/m)*np.sum(logprobs)

    cost = float(np.squeeze(cost))

    return cost

def backward_propagation(parameters,cache,X,Y):

    m = X.shape[1]

    #Extracts weights from passed parameters dictionary
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    #Extracts activation numbers from passed cache dictionary
    A1 = cache["A1"]
    A2 = cache["A2"]

    #Calculation of gradients
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

#Gradient descent
def update_parameters(parameters, grads, learning_rate = 1.2):


    #Extraction of parameter variables
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    #Extraction of gradient variables
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    #Update descent
    W1 = W1-(learning_rate*dW1)
    b1 = b2-(learning_rate*db1)
    W2 = W2-(learning_rate*dW2)
    b2 = b2-(learning_rate*db2)

    #Return new values
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

#Neural network all wrapped up into one function
def nn_model(X,Y,n_h,num_iterations = 10000, print_cost=False):

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    #Training and descent
    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X,parameters)

        cost = compute_cost(A2,Y,parameters)

        grads = backward_propagation(parameters,cache,X,Y)

        parameters = update_parameters(parameters,grads)

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

#Given certain parameters predict the classification of that variable
def predict(parameters,X):

    A2, cache = forward_propagation(X,parameters)
    predictions = A2 > 0.5

    return predictions



#parameters = nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=True)

#plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0])
#plt.title("Decision Boundary for hidden layer size " + str(4))

# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0])
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


plt.show()
