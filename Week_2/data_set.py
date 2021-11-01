

import numpy as np


class LogicalGateData():
    """
    To train the network on logical gates we create for each logical gate a numpy array.
    For And it can look like this:
    and_data = 
    [[1,1,1],
    [1,0,0],
    [0,1,0],
    [0,0,0]]
    where the first and second entry in each item is the input and the last element the expected output (label).
    """

    def get_and_data(self):
        return np.array([[1, 1, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])

    def get_or_data(self):
        return np.array([[1, 1, 1],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0, 0, 0]])

    def get_notand_data(self):
        return np.array([[1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0, 0, 1]])

    def get_notor_data(self):
        return np.array([[1, 1, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

    def get_xor_data(self):
        return np.array([[1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0, 0, 0]])


# to get an example run:
data = LogicalGateData()
print(data.get_notand_data().shape)
