import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from HiddenLayer import HiddenLayer

if __name__ == "__main__": 

    config = {}

    config["learning_rate"] = 0.01

    numHidden = 100
    numLatent = 100
    numInput = 1
    numOutput = 1

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))


    #N x 1
    x = T.matrix()
    #N x 1
    observed_y = T.matrix()

    h1 = HiddenLayer(x, num_in = numInput, num_out = numHidden, initialization = 'xavier', name = "h1")

    z = HiddenLayer(h1.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z')

    h2 = HiddenLayer(z.output, num_in = numLatent, num_out = numHidden, initialization = 'xavier', name = "h2")

    y = HiddenLayer(h2.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', name = "output")

    z_sampled = srng.normal(size = z.shape)

    h2_sampled = HiddenLayer(z_sampled.output, num_in = numLatent, num_out = numHidden, initialization = 'xavier', params = h2.getParams(), name = "h2")

    y_sampled = HiddenLayer(h2_sampled.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', params = y.getParams(), name = "output")

    layers = [h1,z,h2,y]

    params = {}

    for layer in layers: 
        layerParams = layer.getParams()
        for paramKey in layerParams: 
            params[paramKey] = layerParams[paramKey]

    print "params", params

    z_mean = T.mean(z, axis = 1)
    z_var = T.var(z, axis = 1)

    variational_loss = 0.5 * (z_mean + z_var - T.log(z_var))

    loss = T.sum(T.sqr(y.output - observed_y)) + variational_loss

    

    updateObj = Updates(params, loss, config["learning_rate"])

    updates = updateObj.getUpdates()

    train = theano.function(inputs = [x, y_observed], outputs = [y.output, loss], updates = updates)

    sample = theano.function(inputs = [], outputs = [sampled_y])

    for epoch in range(0, config["number_epochs"]): 

        x = np.random.gamma(shape = (100, 1))

        y,loss = train(x,x)

        print "loss", loss





