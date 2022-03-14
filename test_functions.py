"""Test for my functions."""
import numpy as np

from main import simple_neural_network, sigmoid, sigmoid_derivative

def test_sigmoid():

    assert callable(sigmoid)
    assert type(sigmoid(1)) == np.float64
    assert sigmoid(1) == 0.7310585786300049
    assert sigmoid(1.0) == 0.7310585786300049
    
def test_sigmoid_derivative():

    assert callable(sigmoid_derivative)
    assert type(sigmoid_derivative(1)) == int
    assert sigmoid_derivative(1) == 0
    assert sigmoid_derivative(1.0) == 0
    assert sigmoid_derivative(2) == -2
    
    
def test_simple_neural_network():
    
    assert callable(simple_neural_network)
    X=np.array(([0,0,20,3],[0,17,1,2],[6,0,1,5],[1,3,1,1],[1,0,0,1],[3,7,1,0]), dtype=float)
    y=np.array(([0],[0.2],[0],[0.4],[1],[0.6]), dtype=float)
    test_network = simple_neural_network(X,y)
    assert callable(test_network.training)