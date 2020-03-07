import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tools.arff import Arff
import math as math
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate


class Node:
    def __init__(self, attribute_dict, id):
        self.id = id    # Which feature this is
        self.children = []
        self.attribute_dict = attribute_dict


class DTClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,counts=None,target_counts=None):
        self.target_counts=target_counts
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

        num_attributes = len(X[0]) - 1

        # Base Case 1: Pure Node -- All the data is of the same output class
        y_vals = X[:,-1]
        unique_values = np.unique(y_vals)
        if len(unique_values) == 1:
            return unique_values[0]

        # Base Case 3: There are no more features
        if len(counts) == 0:
            return self.find_most_common(X[:,-1])

        entropies = []
        for i in range(num_attributes):
            entropies.append(self.calcEntropy(counts[i], X[:,i], X[:,-1]))   # calcEntropy(self, attribute, col, y):
        if len(entropies) == 0:
            print()

        min_index = np.argmin(entropies)
        node = Node(counts[min_index], attribute_ids[min_index])
        split_data = self.split(min_index, counts[min_index], X)
        del counts[min_index]
        del attribute_ids[min_index]

        for i in range(len(split_data)):
            node.children.append(self.recursiveFit(split_data[i], copy.deepcopy(counts), copy.deepcopy(attribute_ids)))

        return node

    def predict(self, X):
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
        predictions = self.predict(X)
        total = 0
        for i in range(len(y)):
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

def run_sk_learn(X, y):
    d_t_classifier = DecisionTreeClassifier()
    d_t_classifier.fit(X, y)
    score = d_t_classifier.score(X, y)
    return score, d_t_classifier

def seperate_train_and_test():
    # Get the file
    fileName = "zoo.arff"
    print(fileName)
    mat = Arff(fileName, label_count=1)

    # Parse the data
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    X, y = handle_unkowns(data, labels, mat.enum_to_str[len(mat.enum_to_str) - 1])
    counts = mat.enum_to_str
    target_counts = counts[len(counts) - 1]
    counts.pop()
    counts = counts

    PClass = DTClassifier(counts, target_counts)
    PClass.fit(X, y)

    fileName = "all_zoo.arff"
    mat2 = Arff(fileName, label_count=1)
    data2 = mat2.data[:, 0:-1]
    labels2 = mat2.data[:, -1].reshape(-1, 1)
    X2, y2 = handle_unkowns(data2, labels2, mat2.enum_to_str[len(mat2.enum_to_str) - 1])

    score = PClass.score(X2, y2)
    print("My score: ", score)


def start(type=1):
    # Files to be read
    arff_files = [
        "motorcyc_crash_severity"
        # "voting",
        # "pizza",
        # "zoo",
        # "lenses",
        # "all_zoo",
        # "all_lenses",
        # "cars",
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
        counts = mat.enum_to_str
        target_counts = counts[len(counts) - 1]
        counts.pop()
        counts = counts

        if type == 0:
            PClass = DTClassifier(counts, target_counts)
            PClass.fit(X, y)
            score = PClass.score(X, y)
            print("My score: ", score)

        if type == 1:
            PClass2 = DTClassifier(counts, target_counts)
            scores = cross_validate(PClass2, X, y, cv=10)
            sum1 = sum(scores['test_score'])
            sum2 = sum(scores['train_score'])
            avg = sum1 / len(scores['test_score'])
            avg2 = sum2 / len(scores['train_score'])

            print("Test score average: ", avg)
            print("Test score ", scores['test_score'])
            print("Train score average: ", avg2)
            print("Train score ", scores['train_score'])
            print()

        if type == 2:
            d_t_classifier = DecisionTreeClassifier(max_leaf_nodes=10)
            scores = cross_validate(d_t_classifier, X, y, cv=10)
            sum1 = sum(scores['test_score'])
            sum2 = sum(scores['train_score'])
            avg = sum1 / len(scores['test_score'])
            avg2 = sum2 / len(scores['train_score'])

            print("Test score average: ", avg)
            print("Test score ", scores['test_score'])
            print("Train score average: ", avg2)
            print("Train score ", scores['train_score'])
            print()
            # sk_score, clf = run_sk_learn(X, y)
            # tree.export_graphviz(clf, out_file="tree.dot", max_depth=2)
            # # # dot -Tpng tree.dot -o tree.png
            # print(sk_score)

seperate_train_and_test()
start(1)
