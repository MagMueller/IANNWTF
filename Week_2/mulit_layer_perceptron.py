
from preparation import sigmoidprime
from perceptron import Perceptron
import numpy as np
from data_set import LogicalGateData
import matplotlib.pyplot as plt


class MLP():
    # n_perceptrons_in_first_layer, n_hidden_layer,
    def __init__(self, n_inputs, n_perceptrons_per_hidden_layer, n_outputs) -> None:
        self.perceptrons = []

        # hidden layer
        # add 'n_perceptrons_per_hidden_layer' Perceptrons in one layer
        self.perceptrons.append([Perceptron(
            n_inputs) for _ in range(n_perceptrons_per_hidden_layer)])

        # create output neurons
        # for logical gates we have n_outputs = 1
        self.perceptrons.append([Perceptron(
            n_perceptrons_per_hidden_layer) for _ in range(n_outputs)])

    def forward_step(self, inputs):
        """Feed input through network and compute prediction."""
        self.outputs = []

        # appen input layer, is needed for backprob later
        self.outputs.append(inputs)

        # got through all layers
        for layer in range(len(self.perceptrons)):
            # feed in the input in one layer and compute output, which is the input for the next layer
            outputs = np.array(
                [per.forward_step(self.outputs[-1]) for per in self.perceptrons[layer]])
            self.outputs.append(outputs)

    def backprop_step(self, target):
        """Backpropagation of MLP."""
        delta = None
        deltas = {}

        # start in last layer and do backpropagation
        for ind, layer in enumerate(reversed(self.perceptrons)):
            # create an entry for the deltas of the layer
            deltas[len(self.perceptrons)-ind-1] = []

            # go through each Perceptron of the layer
            for n_of_per, per in enumerate(layer):

                # last layer
                if delta is None:
                    # compute delta with target, the output of the last layer, and the derivation of input sum in Perceptron
                    delta = - \
                        (target - self.outputs[-1-ind]) * \
                        sigmoidprime(per.get_input())

                # hidden layers
                else:
                    # Calculate delta: with the last delta, the weights of the next layer that are connected to the current neuron and the derivative of the input signal.
                    delta = np.sum(delta * np.array([next_layer_per.get_weights()[
                                   n_of_per] for next_layer_per in self.perceptrons[-ind]])) * sigmoidprime(per.get_input())  # * np.array(self.outputs[-2-ind])
                # append the deltas of the hole layer to the dict
                deltas[len(self.perceptrons)-ind-1].append(delta)

        # update weights with the computed deltas
        for layer, layer_deltas in deltas.items():
            for per, delta in zip(self.perceptrons[layer], layer_deltas):
                per.update(delta)

    def train(self, data, epochs=100, info_all_n_epochs=100, accuracy_over_last_n_epochs=100):
        """Train your MLP"""
        self.all_loss = []
        self.all_accuracy = []
        self.epochs = epochs
        self.correct = []
        for epoch in range(1, epochs + 1):
            # go through all samples once

            for sample in data:
                # the target is last entry in the sample
                target = sample[2]

                # first to entries are the input data
                self.forward_step(sample[:2])
                self.backprop_step(target)

                self.all_loss.append(self.loss(target))
                self.all_accuracy.append(self.accuracy(
                    target, accuracy_over_last_n_epochs))
            if epoch % info_all_n_epochs == 0:

                print("Epoch {}:".format(epoch))
                print("Loss: {}".format(np.mean(self.all_loss)))
                print("Accuracy: {}".format(np.mean(self.all_accuracy)))
                print("")

    def visualization(self):
        # plot training
        fig, axs = plt.subplots(2)
        axs[0].plot(range(0, self.epochs),
                    self.smooth(np.array(self.all_loss).squeeze(), 10)[::4])
        # axs[2].plot(range(0, self.epochs),
        #            (np.array(self.all_loss).squeeze()[::4]))
        axs[0].set(xlabel='epochs', ylabel='loss',
                   title='Training loss')
        axs[0].grid()

        axs[1].plot(range(0, self.epochs),
                    self.smooth(np.array(self.all_accuracy).squeeze(), 1)[::4])
        axs[1].set(xlabel='epochs', ylabel='accuracy',
                   title='Training accuracy')
        axs[1].grid()
        # fig.savefig("training_data.png")
        plt.show()

    def smooth(self, y, box_pts):
        """Can be used for visualization to make your plot smoother."""
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def loss(self, target):
        return (target - self.outputs[-1])**2

    def accuracy(self, target, accuracy_over_last_n_epochs):
        self.correct.append(np.round(self.outputs[-1]) == target)
        if len(self.correct) <= accuracy_over_last_n_epochs:
            return np.mean(self.correct)
        else:
            return np.mean(self.correct[len(self.correct)-accuracy_over_last_n_epochs:])

    def get_net_information(self):
        return self.perceptrons


# init your mlp for a logical gate
mlp = MLP(n_inputs=2, n_perceptrons_per_hidden_layer=4, n_outputs=1)

# load data, you can check LogicalGateData to see your possibile datasets
data = LogicalGateData().get_or_data()

# train
mlp.train(data, epochs=1000, info_all_n_epochs=100,
          accuracy_over_last_n_epochs=30)
mlp.visualization()
