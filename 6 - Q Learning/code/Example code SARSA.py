#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')
import random

# parameters
gridSize = 4
states_terminal = [[0,0], [gridSize-1, gridSize-1]]
valid_actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
gamma = 0.1 # discount rate
currentReward = -1
numIterations = 100
alpha = 0.1 #exploration factor

# initialization
Q = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

def generateInitialState():
    #generate a random initial state    
    return initialState

def generateNextAction():
    #generate a random action from the valid set of actions
    return nextAction

# define the transition function from a given state and action
def getNextState(state, action):
    
    #define what happens when reaching a terminal state
    if state in states_terminal:
        return ...
    
    # here you should complete this step, the transition step
    nextState = 
   
    # if the agent reaches a wall 
    if -1 in nextState or gridSize in nextState:
        nextState = state
    
    return currentReward, nextState


for it in range(numIterations):
    currentState = generateInitialState()
    
    while True:
        currentAction = generateNextAction()
        reward, nextState = getNextState(currentState, currentAction)
                
        #complete the stop action if the agent reached the goal state
        
        #update the Q-value function Q
        
        #assign as current state the next state
        currentState = nextState    

