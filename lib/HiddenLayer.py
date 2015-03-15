import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng

class HiddenLayer: 

    def __init__(self, input, num_in, num_out, initialization, name, params = None): 

        self.params = params

        if params == None: 

            W_values = np.asarray(0.01 * rng.standard_normal(size=(num_in, num_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name=name + "_W")
        
            b_values = np.zeros((num_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name= name + '_b')

        else: 
            W = params[0]
            b = params[1]

        lin_output = T.dot(input, W) + b

        activation = lambda x: T.maximum(0.0, x)

        self.output = activation(lin_output)

        self.params = [W,b]



    def getParams(self): 
        return self.params



