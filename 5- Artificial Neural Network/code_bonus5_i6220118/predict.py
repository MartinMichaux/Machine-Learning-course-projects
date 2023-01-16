import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

# Useful values
    m = np.shape(X)[0]              #number of examples
    

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
    #transpose given weight matrices
    theta1 = np.transpose(Theta1)
    theta2 = np.transpose(Theta2)

    #add the bias column (column of 1) to the input values (Xs)
    layer1Input = np.hstack((np.ones((m, 1)), X))  
    #calculate the first estimation from the input layer to the hidden one
    firstLayer = sigmoid(np.dot(layer1Input, theta1))
    
    #add the bias column (column of 1) to the input of hidden layer
    layer2Input = np.hstack((np.ones((m,1)), firstLayer))
    #calculate the second estimation from the hidden layer to the output one
    secondLayer = sigmoid(np.dot(layer2Input, theta2))              
                                 
#need to add a +1 term to ensure that the vectors also include the bias unit
    return (np.argmax(secondLayer, axis = 1)+1)

# =========================================================================
