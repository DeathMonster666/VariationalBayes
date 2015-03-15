import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from HiddenLayer import HiddenLayer
from Updates import Updates
import matplotlib.pyplot as plt

theano.config.flaotX = 'float32'

#Network estimates M-dimensional mu, M-dimensional sigma.  
#z = mu + sigma * e
#

if __name__ == "__main__": 

    config = {}

    config["learning_rate"] = 0.0001
    config["number_epochs"] = 20000

    numHidden = 400
    numLatent = 400
    numInput = 1
    numOutput = 1

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))


    #N x 1
    x = T.matrix()
    #N x 1
    observed_y = T.matrix()

    h1 = HiddenLayer(x, num_in = numInput, num_out = numHidden, initialization = 'xavier', name = "h1", activation = "relu")

    z_mean = HiddenLayer(h1.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_mean', activation = None)

    z_var = HiddenLayer(h1.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_var', activation = 'exp')

    z_sampled = srng.normal(size = (100, numLatent))

    z = z_sampled * z_var.output + z_mean.output

    h2 = HiddenLayer(z, num_in = numLatent, num_out = numHidden, initialization = 'xavier', name = "h2", activation = "relu")

    y = HiddenLayer(h2.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', name = "output", activation = None)

    h2_generated = HiddenLayer(z_sampled, num_in = numLatent, num_out = numHidden, initialization = 'xavier', params = h2.getParams(), name = "h2", activation = "relu")

    y_generated = HiddenLayer(h2_generated.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', params = y.getParams(), name = "output", activation = None)

    layers = [h1,z_mean, z_var,h2,y, h2_generated, y_generated]

    params = {}

    for layer in layers: 
        layerParams = layer.getParams()
        for paramKey in layerParams: 
            params[paramKey] = layerParams[paramKey]

    print "params", params


    variational_loss = 0.5 * T.sum(z_mean.output**2 + z_var.output - T.log(z_var.output))

    loss = T.sum(T.sqr(y.output - observed_y)) + variational_loss

    updateObj = Updates(params, loss, config["learning_rate"])

    updates = updateObj.getUpdates()

    train = theano.function(inputs = [x, observed_y], outputs = [y.output, loss, variational_loss, T.mean(z_mean.output), T.mean(z_var.output)], updates = updates)

    print "Finished compiling training function"

    sample = theano.function(inputs = [], outputs = [y_generated.output])

    for epoch in range(0, config["number_epochs"]): 

        x = np.random.normal(loc = 4.0, scale = 2.0, size = (100,1)).astype(np.float32)


        y,loss,variational_loss,mean,var = train(x,x)

        print "x", x[0]
        print "y", y[0][0]
        print "loss", loss
        print "vloss", variational_loss
        print "mean", mean
        print "var", var

    samples = []

    for i in range(0, 1000): 
        samples += sample()[0].tolist()

    print samples
    #plt.hist(samples)
    #plt.show()

    samples = np.asarray(samples)

    print samples.mean()
    print samples.var()


