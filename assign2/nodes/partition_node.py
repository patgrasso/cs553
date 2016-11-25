

import random
from .node import Node

class PartitionNode(Node):

    def __init__(self, trainPercent, validatePercent, testPercent):
        super().__init__()

        if trainPercent + validatePercent + testPercent != 1:
            raise ValueError("""Train, validate, and test percentages do
                not add up to 1""")
        self.trainPercent = trainPercent
        self.validatePercent = validatePercent
        self.testPercent = testPercent

    def __call__(self, data):
        shuffled_data = data
        random.shuffle(shuffled_data)

        n_train = int(self.trainPercent * len(data))
        n_validate = int(self.validatePercent * len(data))
        n_test = int(self.testPercent * len(data))

        # If rounding pushed the sum of sample sizes over the data size,
        # arbitrate the validation sample size
        if n_train + n_validate + n_test != len(data):
            n_validate = len(data) - n_train - n_test

        train_data = shuffled_data[: n_train]
        test_data = shuffled_data[n_train : n_train + n_test]
        validate_data = shuffled_data[n_train + n_test :]

        return {
            'train': train_data,
            'validate': validate_data,
            'test': test_data
        }

