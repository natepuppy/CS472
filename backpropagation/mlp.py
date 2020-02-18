import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import copy
from tools.arff import Arff
from sklearn.linear_model import Perceptron
import scipy.stats as stats
from sklearn.model_selection import train_test_split

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.


# You can get away with doing just the one layer
# The output of a bias node is always one, then it is multiplied by the weight
# There is never an input to a bias node
# input_layer_number does not include bias node

class MLPClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True):
    # def __init__(self, lr=.1, shuffle=True, deterministic=-1):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([2]),  <--- this will create a model with two hidden layers, both 3 nodes wide
            does not include bias node
        """

        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.weights = []
        self.epochs_completed = 0

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.weights = self.initialize_weights(len(X[0])) if not initial_weights else initial_weights
        # self.weights = self.exampleWeights()

        numberOfEpochs = 1000

        for epoch in range(numberOfEpochs):
            for i in range(len(X)):
                input_for_next_layer = X[i]    # This is the input 0, 1, 1  # y[i] is the target
                all_sigmas = []
                all_sigmas.append(copy.deepcopy(X[i]))
                for j in range(len(self.weights)):    # Go through each layer of weights
                    sigmas = []
                    for k in range(len(self.weights[j])):     # Go through each row of the weights for that layer
                        sigmas.append(self.sigmoid(self.multiply_vectors(input_for_next_layer, self.weights[j][k])))    # These are the sigma outputs
                    if j != len(self.weights) - 1: # append bias # don't add to the very last one (the output layer)
                        sigmas.append(1.0)
                    input_for_next_layer = copy.deepcopy(sigmas)
                    all_sigmas.append(input_for_next_layer)

                self.back_propogate(all_sigmas, y[i])


    def back_propogate(self, sigma_values, target):
        new_weights = copy.deepcopy(self.weights)

        previous_deltas = []
        previous_deltas.append(self.calc_output_delta(target, sigma_values[len(sigma_values) - 1][0]))

        for j in reversed(range(len(new_weights))):    # for each layer
            sigmas = sigma_values[j]
            updated_deltas = []
            for k in range(len(new_weights[j])):
                delta = previous_deltas[k]

                for w in range(len(new_weights[j][k])):
                    new_weights[j][k][w] = new_weights[j][k][w] + self.calc_weight_change(self.lr, delta, sigmas[w])

            #previous_deltas = [-0.00548, -0.00548]
            if (j > 0):
                for w in range(len(self.weights[j][0]) - 1):
                    x = 0
                    for k in range(len(self.weights[j])):
                        x = x + (self.weights[j][k][w] * previous_deltas[k])
                    updated_deltas.append(self.calc_hidden_delta(x, sigmas[w]))
                previous_deltas = copy.deepcopy(updated_deltas)
        self.weights = copy.deepcopy(new_weights)





    def calc_output_delta(self, t_1, sigma_1):
        return (t_1 - sigma_1) * (sigma_1) * (1 - sigma_1)

    def calc_hidden_delta(self, prev_delta_times_weight, prev_sigma):
        return prev_delta_times_weight * (prev_sigma * (1 - prev_sigma))

    def calc_weight_change(self, learning_rate, delta, sigma):
        return learning_rate * delta * sigma

    def exampleWeights(self):
        # first iteration
        weights1 = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        weights2 = [[1.0, 1.0, 1.0]]
        return [weights1, weights2]

        # second iteration
        # weights1 = [[1, 1, 1.00113], [1, 1, 1.00113]]
        # weights2 = [[1.00420, 1.00420, 1.00575]]
        # return [weights1, weights2]

    def multiply_vectors(self, a, b):
        total = 0
        for k in range(len(a)):
            total += (a[k] * b[k])
        return total

    def sigmoid_array(self, arr):
        for index in range(len(arr)):
            arr[index] = self.sigmoid(arr[index])
        return arr

    def sigmoid(self, input):
        return 1/(1 + np.exp(-input))

    def initialize_weights(self, length):
        length = 3

        if len(self.hidden_layer_widths) == 0:
            array = np.random.randn(1, 1, length)
            return array

        array = [[] for x in range(len(self.hidden_layer_widths) + 1)]
        # array = np.asarray([])

        for i in range(len(array)):
            if (i == 0):
                array[i] = np.random.randn(self.hidden_layer_widths[i] - 1, length)
            else:
                if i == len(self.hidden_layer_widths):
                    array[i] = np.random.randn(1, self.hidden_layer_widths[i - 1])
                else:
                    array[i] = np.random.randn(self.hidden_layer_widths[i] - 1, self.hidden_layer_widths[i - 1])
        return array

    # Do I need output layers????
    # what about the bias?? -- is it already added into X????
    # can I use a library??
    # def initialize_weights(self, length):
    #     """ Initialize weights for perceptron. Don't forget the bias!  """
    #     layers = []
    #
    #     # If there are no hidden layers just handle it here
    #     if len(self.hidden_layer_widths) == 0:
    #         inputLayerWeights = [[1 for x in range(length)] for y in range(1)]
    #         layers.append(inputLayerWeights)
    #         # self.weights = layers
    #         return layers
    #
    #     # Creates a list containing self.hidden_layer_widths[0] lists, each of length items, all set to 1
    #     inputLayerWeights = [[1 for x in range(length)] for y in range(self.hidden_layer_widths[0] - 1)]
    #     layers.append(inputLayerWeights)
    #
    #     # create the hidden layers
    #     for i in range(len(self.hidden_layer_widths)):
    #         if i == len(self.hidden_layer_widths) - 1:
    #             hiddenLayerWeights = [[1 for x in range(self.hidden_layer_widths[i])] for y in range(1)]
    #             layers.append(hiddenLayerWeights)
    #             break
    #         else:
    #             hiddenLayerWeights = [[1 for x in range(self.hidden_layer_widths[i])] for y in range(self.hidden_layer_widths[i + 1] - 1)]
    #             layers.append(hiddenLayerWeights)
    #     # self.weights = layers
    #     return layers

    def get_weights(self):
        return self.weights
        # if len(self.weights) == 0:
        #     return self.initialize_weights()












PClass = MLPClassifier([5, 12], 1, 0, True)
PClass.fit([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], [1.0, 0.0])  # Both







# Hyper-parameters
learning_rate = 1.0
shuffle = True
split_data = True
training_percentage = .7
sckikitLearn = True


# PClass = MLPClassifier([3, 3], learning_rate, 0, shuffle)
# # PClass.fit([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], [1.0])   # first iteration
# # PClass.fit([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], [0.0])   # second iteration
#
# PClass.fit([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], [1.0, 0.0])  # Both

print("")











# print("self.weights[", j, "][", k, "][", w, "]", self.weights[j][k][w])
# print("previous_deltas[", k, "]", previous_deltas[k])
# print("sigmas[", w, "]", sigmas[w])







# hidden_layer_widths = [3]
# length = 3
#
# if len(hidden_layer_widths) == 0:
#     array = np.random.randn(1, 1, length)
#
# # array = np.array([[np.random.randn(1) for y in range(1)] for x in range(len(hidden_layer_widths) + 1)])
# # array = np.random.randn(1, 1, length)
# # array = np.random.randn(len(hidden_layer_widths) + 1)
#
# array = [[] for x in range(len(hidden_layer_widths) + 1)]
# # array = np.asarray([])
#
#
#
# for i in range(len(array)):
#     if (i == 0):
#         array[i] = np.random.randn(hidden_layer_widths[i] - 1, length)
#     else:
#         if i == len(hidden_layer_widths):
#             array[i] = np.random.randn(1, hidden_layer_widths[i - 1])
#         else:
#             array[i] = np.random.randn(hidden_layer_widths[i] - 1, hidden_layer_widths[i - 1])
#


# array[0].append(np.random.randn(length))
# array = np.random.randn(len(hidden_layer_widths) * length)
# print("2D Array filled with random values : \n", array)
# print("")




# def back_propogate(self, sigma_values, target):
#     previous_deltas = []
#     previous_deltas.append(self.calc_output_delta(target, sigma_values[len(sigma_values) - 1][0]))
#
#     for j in reversed(range(len(self.weights))):
#         sigmas = sigma_values[j]
#         updated_deltas = []
#         for k in range(len(self.weights[j])):
#             delta = previous_deltas[k]
#
#             for w in range(len(self.weights[j][k])):
#                 self.weights[j][k][w] = self.weights[j][k][w] + self.calc_weight_change(1.0, delta, sigmas[w])
#         # previous_deltas = [-0.00548, -0.00548]






def predict(self, X):
    """ Predict all classes for a dataset X
    Args:
        X (array-like): A 2D numpy array with the training data, excluding targets
    Returns:
        array, shape (n_samples,)
            Predicted target values per element in X.
    """


def score(self, X, y):
    """ Return accuracy of model on a given dataset. Must implement own score function.
    Args:
        X (array-like): A 2D numpy array with data, excluding targets
        y (array-like): A 2D numpy array with targets
    Returns:
        score : float
            Mean accuracy of self.predict(X) wrt. y.
    """


def _shuffle_data(self, X, y):
    """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
        It might be easier to concatenate X & y and shuffle a single 2D array, rather than
         shuffling X and y exactly the same way, independently.
    """


# Not required by sk-learn but required by us for grading. Returns the weights.
def get_weights(self):
    return self.weights


def add_bias(self, X):
    biases = np.ones((len(X), 1))
    return np.hstack((X, biases))


# def random_weights(self, length):
#     # Get the total number of neurons that need to be initialized
#     totalNeurons = length
#     for j in range(len(self.hidden_layer_widths)):
#         totalNeurons += self.hidden_layer_widths[j]
#     # generate random numbers with a mean of zero to be the initial weights
#     a, b = -1.0, 1.0
#     mu, sigma = 0.0, .5
#     dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
#     return dist.rvs(totalNeurons)


# Files to be read
arff_files = [
    "seperable",
    # "StandardVoting",
    # "non_seperable",
    # "linsep2nonorigin",
    # "data_banknote_authentication",
]

# Hyper-parameters
learning_rate = .1
shuffle = True
# deterministic = 10  # -1 indicates to run nondeterministically

split_data = True
training_percentage = .7

sckikitLearn = True

for i in range(len(arff_files)):
    # Get the file
    fileName = arff_files[i] + ".arff"
    mat = Arff(fileName, label_count=1)

    # Parse the data
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    PClass = MLPClassifier([3,3], learning_rate, 0, shuffle)
    data = PClass.add_bias(data)
    print(fileName)

    # Run the data either split or not split
    if not split_data:
        PClass.fit(data, labels)
        accuracy = PClass.score(data, labels)
        print("Epochs Completed: ", PClass.epochs_completed)
        print("Accuray = [{:.2f}]".format(accuracy))
        print("Final Weights = ", end='')
        PClass.printWeights()
    else:
        X1, X2, y1, y2 = train_test_split(data, labels, test_size = 0.33)
        PClass.fit(X1, y1)
        train_accuracy = PClass.score(X1, y1)
        test_accuracy = PClass.score(X2, y2)
        print("Epochs Completed: ", PClass.epochs_completed)
        print("Training Accuray = [{:.2f}]".format(train_accuracy))
        print("Test Accuray = [{:.2f}]".format(test_accuracy))
        print("Final Weights = ", end='')
        PClass.printWeights()

# This section runs the voting data on the scikit learn perceptron
fileName = "StandardVoting.arff"
mat = Arff(fileName, label_count=1)
data = mat.data[:, 0:-1]
labels = mat.data[:, -1].reshape(-1, 1)



# import numpy as np
# from sklearn.base import BaseEstimator, ClassifierMixin
#
#
# ### NOTE: The only methods you are required to have are:
# #   * predict
# #   * fit
# #   * score
# #   * get_weights
# #   They must take at least the parameters below, exactly as specified. The output of
# #   get_weights must be in the same format as the example provided.
#
# class MLPClassifier(BaseEstimator, ClassifierMixin):
#
#     def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True):
#         """ Initialize class with chosen hyperparameters.
#
#         Args:
#             hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
#             lr (float): A learning rate / step size.
#             shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
#
#         Example:
#             mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
#         """
#         self.hidden_layer_widths
#         self.lr = lr
#         self.momentum = momentum
#         self.shuffle = shuffle
#
#     def fit(self, X, y, initial_weights=None):
#         """ Fit the data; run the algorithm and adjust the weights to find a good solution
#
#         Args:
#             X (array-like): A 2D numpy array with the training data, excluding targets
#             y (array-like): A 2D numpy array with the training targets
#             initial_weights (array-like): allows the user to provide initial weights
#
#         Returns:
#             self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
#
#         """
#         self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights
#
#         return self
#
#     def predict(self, X):
#         """ Predict all classes for a dataset X
#
#         Args:
#             X (array-like): A 2D numpy array with the training data, excluding targets
#
#         Returns:
#             array, shape (n_samples,)
#                 Predicted target values per element in X.
#         """
#         pass
#
#     def initialize_weights(self):
#         """ Initialize weights for perceptron. Don't forget the bias!
#
#         Returns:
#
#         """
#
#         return [0]
#
#     def score(self, X, y):
#         """ Return accuracy of model on a given dataset. Must implement own score function.
#
#         Args:
#             X (array-like): A 2D numpy array with data, excluding targets
#             y (array-like): A 2D numpy array with targets
#
#         Returns:
#             score : float
#                 Mean accuracy of self.predict(X) wrt. y.
#         """
#
#         return 0
#
#     def _shuffle_data(self, X, y):
#         """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
#             It might be easier to concatenate X & y and shuffle a single 2D array, rather than
#              shuffling X and y exactly the same way, independently.
#         """
#         pass
#
#     ### Not required by sk-learn but required by us for grading. Returns the weights.
#     def get_weights(self):
#         pass
