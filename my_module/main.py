"""Classes used throughout project"""
import numpy as np
import random

def sigmoid(t):
    """
    Take an value and calculate that value on a sigmoid function.
        Sigmoid functions are traditionally functions that go from 0 to 1 where negative numbers are closer to 0
        and positive numbers are closer to 1.
    ------------------------------------------
    Inputs: t
        The value that will be scaled through the sigmoid function.
    -------------------------------------------
    Output:
        returns the value of the sigmoid function at that point.
    """
    return 1 / (1 + np.exp(-t))
    
def sigmoid_derivative(m):
    """"
    Take an value and calculate that value on the derivative of a sigmoid function.
    ------------------------------------------
    Inputs: m
        The value that will be scaled through the derivative of the sigmoid function.
    -------------------------------------------
    Output:
        returns the value of the derivative of the sigmoid function at that point.
    """
    return m * (1 - m)

class simple_neural_network():
    """
    Creates the Neural Network object that will then be trained repeatedly in order to predict a vector based on an input matrix.
    
    For a
    """
    
    def __init__(self, x, y):
        """
        The inputs are as follows:
        X is a matrix with any number of columns but only 4 rows. This in our case is just a random assignment of 0s and 1s but normally
        would be based on some kind of input data such as the darkness of a pixel. This will become our input for the neural network.
        
        Y is a vector with dimension 4 since that is the output we have decided on. It is what we will be comparing our calculated outputs
        to in order to evaluate and train our neural network.
        ----------------------------------------------------
        
        This section just defines the variables to be used elsewhere and creates a base set of weights that are randomly generated.
        """
        
        self.input = x
        self.output = y
        
        #These create sets of random weights with shapes based on the input matrix x and 4 which is what the output of this neural network will be
        self.weights1 = np.random.rand(self.input.shape[1],self.output.shape[0])
        self.weights2 = np.random.rand(self.output.shape[0],1)
        
    def training(self, epochs = 1000):
        """
        The function that trains the neural network.
        --------------------------------------------
        
        The input epochs indicates how many times the network will be trained, a base of 1000 is used. Typically networks need a high
        number of epochs so when training this pick a large number to use if you would like the end result to be accurate.
        
        --------------------------------------------
        This function will train the neural network and print out the evaluation of it every so often and then again at the end of training.
        Note that the function does not redifine the variables so you can train it repeatedly and it will improve the process but the 
        epoch counter will start over at the number that is entered every time.
        """
        
        self.epochs = epochs
        
        for i in range(self.epochs):
            
            #This section multiplies the weights by the input and uses the sigmoid function to turn those weights back to numbers 0-1
            self.l1 = sigmoid(np.dot(self.input, self.weights1))
            self.l2 = sigmoid(np.dot(self.l1, self.weights2))
            
            self.l2_error = self.output - self.l2
            self.meansquaredloss = np.mean(np.square(self.l2_error))
            
            #This is just a set of print statements to print out how well the network performs every 200 epochs
            if i % 200 == 0:
                
                print(f'Epoch: \n{i}')
                print(f'Output: \n{self.l2}')
                print(f'Error: \n{self.l2_error}')
                print(f'Mean Squared Loss: \n{self.meansquaredloss}')
                print('\n')
                
            #This is backpropagation, using the error we calculated above we change the weights based on the delta value which is how wrong it was
            self.delta_weights2 = np.dot(self.l1.T, self.l2_error*sigmoid_derivative(self.l2))
            self.l1_error = np.dot((self.output - self.l2)*sigmoid_derivative(self.l2), self.weights2.T)
            self.delta_weights1 = np.dot(self.input.T, self.l1_error*sigmoid_derivative(self.l1))
            
            """
            The math in this section may not be entirely clear so I'll try to explain what's going on somewhat.
            The two variables that are delta are how much we will change our original weights on. The original
            weights are completely random and so based on how wrong we are we try to correct that. To do so we
            take how active the neurons in the previous step were, or the value of the sigmoid function, and 
            multiply that with another term, which is found from the error in the previous step times the derivative
            of the sigmoid function. To do so we use a dot product because these are vectors of all the individual
            neurons in the set not singular values.
            """
            self.weights1 += self.delta_weights1
            self.weights2 += self.delta_weights2
                
        print(f'Final Predicted Values: \n{self.l2}')
        print(f'\nOriginal Output: \n{self.output}')
