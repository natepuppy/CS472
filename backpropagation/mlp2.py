# import numpy as np
# from sklearn.base import BaseEstimator, ClassifierMixin
# import copy
# from tools.arff import Arff
# from sklearn.linear_model import Perceptron
# import scipy.stats as stats
# from sklearn.model_selection import train_test_split
#
# ### NOTE: The only methods you are required to have are:
# #   * predict
# #   * fit
# #   * score
# #   * get_weights
# #   They must take at least the parameters below, exactly as specified. The output of
# #   get_weights must be in the same format as the example provided.
#
# class MLPClassifier(BaseEstimator,ClassifierMixin):
#     def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True):
#     # def __init__(self, lr=.1, shuffle=True, deterministic=-1):
#         """ Initialize class with chosen hyperparameters.
#
#         Args:
#             hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
#             lr (float): A learning rate / step size.
#             shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
#
#         Example:
#             mlp = MLPClassifier([2]),  <--- this will create a model with two hidden layers, both 3 nodes wide
#             does not include bias node
#         """
#
#     # You can get away with doing just the one layer
#
#         self.hidden_layer_widths = hidden_layer_widths
#         self.lr = lr
#         self.momentum = momentum
#         self.shuffle = shuffle
#         self.weights = []
#         self.epochs_completed = 0
#
#     def fit(self, X, y, initial_weights=None):
#         """ Fit the data; run the algorithm and adjust the weights to find a good solution
#         Args:
#             X (array-like): A 2D numpy array with the training data, excluding targets
#             y (array-like): A 2D numpy array with the training targets
#             initial_weights (array-like): allows the user to provide initial weights
#         Returns:
#             self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
#
#         """
#         # self.weights = self.initialize_weights(len(X[0])) if not initial_weights else initial_weights
#         self.weights = self.initialize_weights(len(X[0])) if not initial_weights else initial_weights
#
#         self.weights = self.exampleWeights()
#
#         numberOfEpochs = 1
#
#         for epoch in range(numberOfEpochs):
#             for i in range(len(X)):
#                 input_for_next_layer = X[i]    # This is the input 0, 1, 1  # y[i] is the target
#                 all_sigmas = []
#                 all_sigmas.append(copy.deepcopy(X[i]))
#                 for j in range(len(self.weights)):    # Go through each layer of weights
#                     sigmas = []
#                     for k in range(len(self.weights[j])):     # Go through each row of the weights for that layer
#                         sigmas.append(self.sigmoid(self.multiply_vectors(input_for_next_layer, self.weights[j][k])))    # These are the sigma outputs
#                     if j != len(self.weights) - 1: # append bias # don't add to the very last one (the output layer)
#                         sigmas.append(1)
#                     input_for_next_layer = copy.deepcopy(sigmas)
#                     all_sigmas.append(input_for_next_layer)
#
#                 self.back_propogate(all_sigmas, y)
#
# # The output of a bias node is always one, then it is multiplied by the weight
# # There is never an input to a bias node
# # input_layer_number does not include bias node
#
#     def back_propogate(self, sigma_values, targets):
#         deltas = []
#         for j in reversed(range(len(self.weights))):
#             for k in range(len(self.weights[j])):
#                 sigmas = sigma_values[j + 1]
#                 if j == len(self.weights) - 1:   # If this is the last layer of weights
#                     deltas.append(self.calc_output_delta(targets[i], sigma_values[len(sigma_values) - 1][0]))
#                 else:
#
#                 for w in range(len(self.weights[j][k])):
#                     pass
#
#
#
#
#
#
#
#
#
#
#
#
#     def calc_output_delta(self, t_1, sigma_1):
#         return (t_1 - sigma_1)(sigma_1)(1 - sigma_1)
#
#     def calc_hidden_delta(self, prev_delta, weight, prev_sigma):
#         return (prev_delta)(weight)(prev_sigma(1 - prev_sigma))
#
#     def calc_weight_change(self, learning_rate, delta, sigma):
#         return learning_rate * delta * sigma
#
#
#
#
#     def exampleWeights(self):
#         weights1 = [[1, 1, 1.00113], [1, 1, 1.00113]]
#         weights2 = [[1.00420, 1.00420, 1.00575]]
#         return [weights1, weights2]
#
#     def multiply_vectors(self, a, b):
#         total = 0
#         for k in range(len(a)):
#             total += (a[k] * b[k])
#         return total
#
#     def sigmoid_array(self, arr):
#         for index in range(len(arr)):
#             arr[index] = self.sigmoid(arr[index])
#         return arr
#
#     def sigmoid(self, input):
#         return 1/(1 + np.exp(-input))
#
#
#
#
#
#
#     # Do I need output layers????
#     # what about the bias?? -- is it already added into X????
#     # can I use a library??
#     def initialize_weights(self, length):
#         """ Initialize weights for perceptron. Don't forget the bias!  """
#         layers = []
#
#         # If there are no hidden layers just handle it here
#         if len(self.hidden_layer_widths) == 0:
#             inputLayerWeights = [[1 for x in range(length)] for y in range(1)]
#             layers.append(inputLayerWeights)
#             # self.weights = layers
#             return layers
#
#         # Creates a list containing self.hidden_layer_widths[0] lists, each of length items, all set to 1
#         inputLayerWeights = [[1 for x in range(length)] for y in range(self.hidden_layer_widths[0] - 1)]
#         layers.append(inputLayerWeights)
#
#         # create the hidden layers
#         for i in range(len(self.hidden_layer_widths)):
#             if i == len(self.hidden_layer_widths) - 1:
#                 hiddenLayerWeights = [[1 for x in range(self.hidden_layer_widths[i])] for y in range(1)]
#                 layers.append(hiddenLayerWeights)
#                 break
#             else:
#                 hiddenLayerWeights = [[1 for x in range(self.hidden_layer_widths[i])] for y in range(self.hidden_layer_widths[i + 1] - 1)]
#                 layers.append(hiddenLayerWeights)
#         # self.weights = layers
#         return layers
#
#     def get_weights(self):
#         return self.weights
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # Hyper-parameters
# learning_rate = .1
# shuffle = True
# split_data = True
# training_percentage = .7
# sckikitLearn = True
#
#
# PClass = MLPClassifier([3], learning_rate, 0, shuffle)
# PClass.fit([[0, 1, 1], [0, 0, 1]], [], )
#
#
#
#
#
#
#
#
#
