import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
from HiddenLayer import HiddenLayer
from Updates import Updates
import matplotlib.pyplot as plt
from RandomData import getData
import math

'''
Given an input vector X and an output vector Y, model the conditional distribution p(Y | X).  

Before variational layer, use both X and Y.  

After variational layer, use only X.  

'''

theano.config.flaotX = 'float32'


if __name__ == "__main__": 

    config = {}

    config["learning_rate"] = 0.0001
    config["number_epochs"] = 200000
    config["report_epoch_ratio"] = 400
    config["popups"] = True

    x_size = 1
    y_size = 1

    numHidden = 800
    numLatent = 800
    numInput = x_size + y_size
    numOutput = y_size

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

    join = lambda a,b: T.concatenate([a,b], axis = 0)

    #N x 1
    y = T.matrix()

    #N x 1
    x = T.matrix()

    #N x 1
    observed_y = T.matrix()

    h1 = HiddenLayer(join(x,y), num_in = numInput, num_out = numHidden, initialization = 'xavier', name = "h1", activation = "relu")

    h2 = HiddenLayer(h1.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', name = "h2", activation = "relu")

    z_mean = HiddenLayer(h2.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_mean', activation = None)

    z_var = HiddenLayer(h2.output, num_in = numHidden, num_out = numLatent, initialization = 'xavier', name = 'z_var', activation = 'exp')

    z_sampled = srng.normal(size = (100, numLatent))

    z = z_sampled * z_var.output + z_mean.output

    #At train time, we use q(z | y).  
    h3 = HiddenLayer(join(z,x), num_in = numLatent + x_size, num_out = numHidden, initialization = 'xavier', name = "h3", activation = "relu")

    h4 = HiddenLayer(h3.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', name = "h4", activation = "relu")

    y_train = HiddenLayer(h4.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', name = "output", activation = None)

    #To sample, we use the prior p(z) for q(z | y).  

    z_prior = z_sampled

    h3_generated = HiddenLayer(join(z_prior,x), num_in = numLatent + x_size, num_out = numHidden, initialization = 'xavier', params = h3.getParams(), name = "h3", activation = "relu")

    h4_generated = HiddenLayer(h3_generated.output, num_in = numHidden, num_out = numHidden, initialization = 'xavier', params = h4.getParams(), name = "h4", activation = "relu")

    y_generated = HiddenLayer(h4_generated.output, num_in = numHidden, num_out = numOutput, initialization = 'xavier', params = y_train.getParams(), name = "output", activation = None)

    layers = [h1,z_mean,z_var,h2,h3,y_train,h4]

    params = {}

    for layer in layers: 
        layerParams = layer.getParams()
        for paramKey in layerParams: 
            params[paramKey] = layerParams[paramKey]

    print "params", params


    variational_loss = 0.5 * T.sum(z_mean.output**2 + z_var.output - T.log(z_var.output) - 1.0)

    loss = T.sum(T.sqr(y_train.output - observed_y)) + variational_loss

    updateObj = Updates(params, loss, config["learning_rate"])

    updates = updateObj.getUpdates()

    train = theano.function(inputs = [x, y, observed_y], outputs = [y_train.output, loss, variational_loss, T.mean(z_mean.output), T.mean(z_var.output)], updates = updates)

    print "Finished compiling training function"

    sample = theano.function(inputs = [x], outputs = [y_generated.output])

    lossLst = []
    xLst = getData(size = (1000)).tolist()

    for epoch in range(0, config["number_epochs"]): 

        x = getData(size = (100,1))

        y,loss,variational_loss,mean,var = train(x,x)

        lossLst += [math.log(loss)]

        if epoch % config["report_epoch_ratio"] == 0: 

            print "x", x[0]
            print "y", y[0][0]
            print "loss", loss
            print "vloss", variational_loss
            print "mean", mean
            print "var", var

            samples = []

            for i in range(0, 10): 
                samples += sample()[0].tolist()
                

            samples = np.asarray(samples)

            print "sample mean", samples.mean()
            print "sample p50", np.percentile(samples, 50.0)
            print "sample p90", np.percentile(samples, 90.0)
            print "true mean", np.asarray(xLst).mean()
            print "true p50", np.percentile(np.asarray(xLst), 50.0)
            print "true p90", np.percentile(np.asarray(xLst), 90.0)

            if config["popups"]:
                bins = np.arange(-2.0, 20.0, 2.0)
                plt.hist(xLst, alpha = 0.5, bins = bins)
                plt.hist(samples, alpha = 0.5, bins = bins)
                plt.show()

                plt.plot(lossLst)
                plt.show()

