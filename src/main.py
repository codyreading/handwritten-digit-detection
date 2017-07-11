#!/usr/bin/env python

"""
main.py
~~~~~~~

Runs the training of the neural network on the MNIST database. The number of neurons
in each layer and the number of layers can be adjusted, as well as the learning rate, 
the mini batch size, and the number of epochs to train for. 
"""

# Libraries
import mnist_loader
import network

# Parameters
LAYERS = [784, 30, 10] # LAYERS = [# of Neurons in first layer, # of Neurons in
                       #            second layer, ...]
NUM_EPOCHS = 30
LEARNING_RATE = 10
MINI_BATCH_SIZE = 3.0

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network(LAYERS)
    net.SGD(training_data, NUM_EPOCHS, LEARNING_RATE, MINI_BATCH_SIZE, test_data=test_data)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as err:
        print(err.args[0])
