
from preparation import sigmoidprime
from perceptron import Perceptron
import numpy as np
from data_set import LogicalGateData
import matplotlib.pyplot as plt


class MLP():
    # n_perceptrons_in_first_layer, n_hidden_layer,
    def __init__(self, n_inputs, n_perceptrons_per_hidden_layer, n_outputs) -> None:
        self.perceptrons = []

        # first layer
        # for perceptron in n_perceptrons_in_first_layer:
        #    perceptrons[0, perceptron] = Perceptron(n_inputs)

        # hidden layers
        # for layer in range(n_hidden_layer):
        self.perceptrons.append([Perceptron(
            n_inputs) for _ in range(n_perceptrons_per_hidden_layer)])
        # for ind in range(n_perceptrons_per_hidden_layer):
        #    self.perceptrons[0, ind] = Perceptron(
        #        n_inputs)

        # create output neurons, for logical gates we have n_outputs = 1
        self.perceptrons.append([Perceptron(
            n_perceptrons_per_hidden_layer) for _ in range(n_outputs)])
        # for ind in range(n_outputs):
        #    self.perceptrons[1, ind] = Perceptron(
        #        n_perceptrons_per_hidden_layer)

    def forward_step(self, inputs):

        self.outputs = []
        self.outputs.append(inputs)

        for ind in range(len(self.perceptrons)):
            # feed in input and compute output, which is the input for the next layer
            outputs = np.array(
                [per.forward_step(self.outputs[-1]) for per in self.perceptrons[ind]])
            self.outputs.append(outputs)
        # self.inputs.append(inputs)
        # store the output of the last layer
        # self.prediction = self.inputs
        # return the output of the last layer
        # return self.inputs

    def backprop_step(self, target):
        delta = None
        deltas = {}
        for ind, layer in enumerate(reversed(self.perceptrons)):
            deltas[len(self.perceptrons)-ind-1] = []
            for n_of_per, per in enumerate(layer):
                # last layer
                if delta is None:
                    delta = - \
                        (target - self.outputs[-1-ind]) * \
                        sigmoidprime(per.get_input())  # * self.outputs[-2-ind]

                # hidden layers
                else:
                    delta = np.sum(delta * np.array([next_layer_per.get_weights()[
                                   n_of_per] for next_layer_per in self.perceptrons[-ind]])) * sigmoidprime(per.get_input())  # * np.array(self.outputs[-2-ind])
                deltas[len(self.perceptrons)-ind-1].append(delta)

        # update weights with computed deltas
        for layer, layer_deltas in deltas.items():
            for per, delta in zip(self.perceptrons[layer], layer_deltas):
                per.update(delta)

    """
        deltas = None
        for layer in reversed(self.perceptrons):
            for per in layer:
                # last layer
                if deltas is None:
                    deltas = -(target - per.get_output())
                    per.update(delta)
                # hidden layer
                else:
                    deltas = deltas *

        # hidden layer
        for per in perceptrons[0]:
            delta = -(target - self.output)
    """

    def train(self, data, epochs=100, info_all_n_epochs=100):
        self.all_loss = []
        self.all_accuracy = []
        self.epochs = epochs
        for epoch in range(1, epochs + 1):
            for sample in data:
                target = sample[2]
                self.forward_step(sample[:2])
                self.backprop_step(target)
                self.all_loss.append(self.loss(target))
                self.all_accuracy.append(self.accuracy(target))
            if epoch % info_all_n_epochs == 0:

                print("Epoch {}:".format(epoch))
                print("Loss: {}".format(np.mean(self.all_loss)))
                print("Accuracy: {}".format(np.mean(self.all_accuracy)))
                print("")

    def visualization(self):
        # plot training
        fig, axs = plt.subplots(2)
        axs[0].plot(range(0, self.epochs),
                    np.array(self.all_loss).squeeze()[::4])
        axs[0].set(xlabel='epochs', ylabel='loss',
                   title='Training loss')
        axs[0].grid()

        axs[1].plot(range(0, self.epochs),
                    np.array(self.all_accuracy).squeeze()[::4])
        axs[1].set(xlabel='epochs', ylabel='accuracy',
                   title='Training accuracy')
        axs[1].grid()
        # fig.savefig("training_data.png")
        plt.show()

    def loss(self, target):
        return (target - self.outputs[-1])**2

    def accuracy(self, target):
        return np.round(self.outputs[-1]) == target

    def get_net_information(self):
        return self.perceptrons


m = MLP(n_inputs=2, n_perceptrons_per_hidden_layer=4, n_outputs=1)
data = LogicalGateData()
m.train(data.get_xor_data(), 10000)
m.visualization()
