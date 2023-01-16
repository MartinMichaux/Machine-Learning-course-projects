import numpy as np

def randInitializeWeights(L_in, L_out, interval):
#RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
#incoming connections and L_out outgoing connections
#   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
#   of a layer with L_in incoming connections and L_out outgoing 
#   connections. 
#
#   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
#   the first row of W handles the "bias" terms
#

# You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

# ====================== YOUR CODE HERE ======================
# Instructions: Initialize W randomly so that we break the symmetry while
#               training the neural network.
#
# Note: The first row of W corresponds to the parameters for the bias units
#

    #get random value from a uniform distribution in the given interval
    W = np.random.uniform(low=-interval, high=interval, size=(L_out,1 + L_in)) 


    
# =========================================================================

    return W
