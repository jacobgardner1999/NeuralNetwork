# Readme - Neural network tutorial

This project follows the Neural Networks and Deep Learning book by Michael Nielson.
http://neuralnetworksanddeeplearning.com/index.html
https://github.com/unexploredtest/neural-networks-and-deep-learning

I have annotated the algorithms as I have learnt about them from the book and taken notes in Obsidian 
which I will add as a folder.

The code is all taken straight from the book and can be run in a python shell as follows:

```
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
net = network.Network([784,30,10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```

