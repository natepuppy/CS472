import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tools.arff import Arff
from sklearn.model_selection import train_test_split
import math as math
import copy
from sklearn import tree

# TODO:
# 1: Implement the score and
# 2: Implement predict (10 fold Cross Validation)
# 3: Print the tree using export_graphviz
# 5: Test on evaluation data, create evaluation.csv
# 5: Implement the SK decision tree



class Node:
    def __init__(self, attribute_dict, id, is_leaf):
        self.id = id    # Which feature this is
        self.children = []
        self.attribute_dict = attribute_dict
        self.is_leaf = is_leaf



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
        # Base Case 1: If there is no more data
        if len(X) == 0:
            return 0.0

        # Base Case 1: Pure Node -- All the data is of the same output class
        y_vals = X[:,(len(X[0]) - 1)]
        unique_values = np.unique(y_vals)
        if len(unique_values) == 1:
            return unique_values[0]

        # Base Case 3: There are no more features
        if len(counts) == 0:
            return self.find_most_common(X[:,(len(X[0]) - 1)])

        entropies = []
        for i in range(len(X[0]) - 1):
            entropies.append(self.calcEntropy(counts[i], X[:,i], X[:,len(X[0])-1]))
        min_index = np.argmin(entropies)
        node = Node(counts[min_index], attribute_ids[min_index], False)
        split_data = self.split(min_index, counts[min_index], X)

        del counts[min_index]
        del attribute_ids[min_index]

        for i in range(len(split_data)):
            node.children.append(self.recursiveFit(split_data[i], copy.deepcopy(counts), copy.deepcopy(attribute_ids)))

        return node

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        predictions = []
        for i in range(len(X)):
            predictions.append(self.predict_one(X[i]))
        return predictions


    def predict_one(self, X_i):
        node = self.tree
        while True:
            if isinstance(node, float) or isinstance(node, int):
                return node

            attribute_class = node.id
            attribute_instance_value = X_i[attribute_class]
            node = node.children[int(attribute_instance_value)]










    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """
        predictions = self.predict(X)
        total = 0
        for i in range(len(y)):
            # print(y[i][0], " ", predictions[i])
            if abs(y[i][0] - predictions[i]) < .001:
                total += 1
            else:
                total += 0

        return total / len(y)










    def find_most_common(self, arr):
        if len(arr) == 0:
            return 0.0
        counts = []
        for j in range(len(self.target_counts)):   # for each output class
            total_instances = 0
            for i in range(len(arr)):    # For each output
                if arr[i] == self.target_counts[j]:
                    total_instances += 1
            counts.append(total_instances)
        x = np.argmax(counts)
        return float(x)


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
    # X, y = delete_if_over_half_are_missing(X, y)
    means = get_mean_foreach_output_class(X, y, output_classes)

    num_instances = len(X)
    num_attributes = len(X[0])

    for k in range(len(output_classes)):        # bad, good, best
        for i in range(num_attributes):         # for each attribute  # meat, veg, crust
            for j in range(num_instances):      # for each instance  # 1...n
                if (math.isnan(X[j][i])) and y[j][0] == k:
                    X[j][i] = means[i][k]

    return X, y



    # for i in range(num_attributes):   # for each attribute
    #     total_real_instances = 0
    #     total = 0
    #     for j in range(num_instances):    # for each instance
    #         if not math.isnan( X[j][i] ):
    #             total += X[j][i]
    #             total_real_instances += 1
    #     mean = total / total_real_instances
    #     all_means.append(mean)




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

def get_mean_foreach_output_class(X, y, output_classes):
    num_instances = len(X)
    num_attributes = len(X[0])

    avg_values = [[0 for i in range(len(output_classes))] for j in range(len(X[0]))]     # number_of_attributes x num_output_classes

    for k in range(len(output_classes)):        # bad, good, best
        for i in range(num_attributes):         # for each attribute  # meat, veg, crust
            total_real_instances = 0
            total = 0
            for j in range(num_instances):      # for each instance  # 1...n
                if (not math.isnan(X[j][i])) and y[j][0] == k:
                    total += X[j][i]
                    total_real_instances += 1
            mean = total / total_real_instances
            avg_values[i][k] = mean
    return avg_values


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
        PClass.fit(X, y)

        score = PClass.score(X, y)
        print(score)




def seperate_train_and_test():
    mat = Arff("zoo.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    # X, y = handle_unkowns(data, labels, mat.enum_to_str[len(mat.enum_to_str) - 1])

    # Create and run classifier
    PClass = DTClassifier(counts=mat.enum_to_str)
    PClass.fit(data, labels)

    mat2 = Arff("all_zoo.arff", label_count=1)
    data2 = mat2.data[:, 0:-1]
    labels2 = mat.data[:, -1].reshape(-1, 1)
    # X, y = handle_unkowns(data, labels, mat.enum_to_str[len(mat.enum_to_str) - 1])
    pred = PClass.predict(data2)
    np.savetxt("pred_zoo.csv", pred, delimiter=",")
    score = PClass.score(data2, labels2)
    print(score)



seperate_train_and_test()
start()





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





