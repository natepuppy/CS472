import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tools.arff import Arff
from sklearn.model_selection import train_test_split
import math as math
import copy



class Node:
    def __init__(self, attribute_dict, id):
        self.id = id
        self.children = []
        self.attribute_dict = attribute_dict



class DTClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,counts=None):
        self.target_counts = counts[len(counts) - 1]
        counts.pop()
        self.counts = counts
        self.tree = None


    def fit(self, X, y):
        X = np.hstack((X, y))
        attribute_ids = [i for i in range(len(self.counts))]
        self.tree = self.recursiveFit(X, self.counts, attribute_ids)
        return self

    def recursiveFit(self, X, counts, attribute_ids):    # What is len X is zero?
        if len(X) == 0:
            return None    # What do I do here???
        entropies = []
        for i in range(len(X[0]) - 1):
            entropies.append(self.calcEntropy(counts[i], X[:,i], X[:,len(X[0])-1]))
        min_index = np.argmin(entropies)
        node = Node(counts[min_index], attribute_ids[min_index])
        split_data = self.split(min_index, counts[min_index], X)
        del counts[min_index]
        del attribute_ids[min_index]

        # If there is only one attribute left that hasn't been used to decide,
        # put that node as the child, and return the last one that was split on
        if len(counts) == 1:
            child = Node(counts[0], attribute_ids[0])
            node.children.append(child)
            return node

        for i in range(len(split_data)):
            node.children.append(self.recursiveFit(split_data[i], copy.deepcopy(counts), copy.deepcopy(attribute_ids)))

        return node


    def calcEntropy(self, attribute, col, y):   # attribute = dict = {0: 'N', 1: 'Y'}
        num_instances = len(col)
        calculated_values = [[0 for i in range(len(self.target_counts) + 1)] for j in range(len(attribute))]

        for j in range(len(col)):
            calculated_values[int(col[j])][int(y[j])] += 1
            calculated_values[int(col[j])][len(self.target_counts)] += 1

        total_entropy = 0
        for j in range(len(calculated_values)):
            if calculated_values[j][len(self.target_counts)] == 0:
                continue
            ratio = calculated_values[j][len(self.target_counts)] / num_instances
            log_values = 0
            for k in range(len(calculated_values[j]) - 1):   # adjust for the total that is at the end of the array
                ratio2 = (calculated_values[j][k] / calculated_values[j][len(self.target_counts)])
                if ratio2 == 0:
                    continue
                log_values += (-1 * ratio2 * math.log2(ratio2))
            total_entropy += ratio * log_values
        return total_entropy

    def split(self, index, attribute, X):
        x_copy = copy.deepcopy(X)
        split_data = [[] for i in range(len(attribute))]

        for j in range(len(x_copy)):    # For every instance
            split_data[int(x_copy[j][index])].append(x_copy[j])

        for j in range(len(split_data)):
            if len(split_data[j]) > 0:
                split_data[j] = np.delete(split_data[j], index, 1)

        return split_data

def handle_unkowns(X, y, output_classes):
    X, y = delete_if_over_half_are_missing(X, y)

    x = np.mean(X, axis=0)
    print()

    # # create a 2d array here to hold all of these values
    # avg_values = [[0 for i in range(len(output_classes))] for j in range(len(X))]
    # for i in range(len(X[0])):    # for each attribute
    #     for j in range(len(output_classes)):   # for each output class



# delete if over half the attributes are missing, else, keep them and handle them in a different way
def delete_if_over_half_are_missing(X, y):
    num_attributes = len(X[0])
    i = 0
    for row in X:
        num_missing = 0
        for j in range(len(row)):
            if math.isnan(row[j]):
                num_missing += 1
        percent_missing = num_missing / num_attributes
        if percent_missing > .50:
            X = np.delete(X, i, 0)
            y = np.delete(y, i, 0)
            i -= 1
        i += 1
    return X, y




def start():
    # Files to be read
    arff_files = [
        "voting",
        "pizza",
        "zoo",
        "lenses",
        "all_zoo",
        "all_lenses",
    ]

    for i in range(len(arff_files)):
        # Get the file
        fileName = arff_files[i] + ".arff"
        print(fileName)
        mat = Arff(fileName,label_count=1)

        # Parse the data
        data = mat.data[:,0:-1]
        labels = mat.data[:,-1].reshape(-1,1)

        X, y = handle_unkowns(data, labels, mat.enum_to_str[len(mat.enum_to_str) - 1])

        # Create and run classifier
        PClass = DTClassifier(counts=mat.enum_to_str)
        PClass.fit(data, labels)




start()






### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score


# def predict(self, X):
#     """ Predict all classes for a dataset X
#     Args:
#         X (array-like): A 2D numpy array with the training data, excluding targets
#     Returns:
#         array, shape (n_samples,)
#             Predicted target values per element in X.
#     """
#     pass
#
#
# def score(self, X, y):
#     """ Return accuracy of model on a given dataset. Must implement own score function.
#
#     Args:
#         X (array-like): A 2D numpy array with data, excluding targets
#         y (array-li    def _shuffle_data(self, X, y):
#     """
#     return 0
#
# # Get the number of instances, number of attributes, all the attributes and the number possible values for each instance.
#


""" Initialize class with chosen hyperparameters.
Args:
    counts = how many types for each attribute
Example:
    DT  = DTClassifier()
"""

""" Fit the data; Make the Desicion tree
Args:
    X (array-like): A 2D numpy array with the training data, excluding targets
    y (array-like): A 2D numpy array with the training targets
Returns:
    self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
"""





