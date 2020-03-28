import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tools.arff import Arff
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA


class KNNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,columntype=[],weight_type='inverse_distance',k_value=3,dist_metric=False,ranges=[]):
        self.columntype = columntype
        self.weight_type = weight_type
        self.k_value = k_value
        self.dist_metric = dist_metric
        self.ranges = ranges

    def fit(self,X,y):
        self.X = X
        self.y = y
        return self

    def predict(self, data):
        return_values = []

        for i in range(len(data)):
            distances = []
            indexes = []
            outputs = []
            for j in range(len(self.X)):   # each instance in the train set
                dist = self.calculate_distance(data[i], self.X[j])

                if len(distances) < self.k_value:
                    distances.append(dist)
                    indexes.append(j)
                    outputs.append(self.y[j])
                else:
                    if self.weight_type == 'inverse_distance':
                        min_value_index = int(np.argmin(distances))
                        if distances[min_value_index] < dist:
                            distances[min_value_index] = dist
                            indexes[min_value_index] = j
                            outputs[min_value_index] = self.y[j]
                    else:
                        max_value_index = int(np.argmax(distances))
                        if distances[max_value_index] > dist:
                            distances[max_value_index] = dist
                            indexes[max_value_index] = j
                            outputs[max_value_index] = self.y[j]
            # now that we have the k closest instances, find the correct output class, or regression value
            if self.columntype[-1] != 'nominal':
                output = self.determine_regression_prediction(distances, indexes, outputs)
            else:
                output = self.determine_nominal_class(distances, outputs)
            return_values.append(output)
        return return_values

    def calculate_distance(self, arr1, arr2):
        total = 0
        for k in range(len(arr1)):
            if arr1[k] == '?' or arr2[k] == '?':
                total += 1
                continue
            if self.columntype[k] == 'nominal':
                if arr1[k] != arr2[k]:
                    total = 1
            else:
                if not self.dist_metric:
                    total += (arr1[k] - arr2[k]) ** 2
                else:
                    total += np.abs((arr1[k] - arr2[k])) / self.ranges[k]

        # runs faster with this code, but it doesn't work for heterogeneous data
        # dist = 0
        # if self.columntype == 'nominal':  # find distance for two nominal values
        #     if arr1[k] != arr2[k]:
        #         dist = 1
        # else:
        #     dist = (arr1[k] - arr2[k]) ** 2
        # total += dist

        total = np.sqrt(total)
        if self.weight_type == 'inverse_distance':
            if total == 0:
                return 1   # FIXME!!!
            total = 1 / (total ** 2)
        return total

    def determine_nominal_class(self, distances, outputs):
        if self.weight_type == 'inverse_distance':
            vals = []
            values, counts = np.unique(outputs, return_counts=True)
            for k in range(len(values)):   # for each class
                vals.append(0)
                for a in range(len(distances)): # for each values
                    if outputs[a] == values[k]:
                        vals[k] += distances[a]
            ind = np.argmax(vals)
            return values[ind]

        values, counts = np.unique(outputs, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    def determine_regression_prediction(self, distances, indexes, outputs):
        numerator = 0
        denominator = 0
        for k in range(len(indexes)):
            index = indexes[k]
            numerator += self.y[index] * distances[k]
            denominator += distances[k]
        return numerator / denominator

    def score(self, X, y):
        predictions = self.predict(X)

        if self.columntype[-1] == 'nominal':
            total = 0
            for i in range(len(y)):
                if abs(y[i][0] - predictions[i]) < .0000000001:
                    total += 1
                else:
                    total += 0
            return total / len(y)
        else:
            total = 0
            for i in range(len(y)):
                total += ((y[i][0] - predictions[i]) ** 2)
            return (total / len(y))

def wrapper(X, y):
    num_features_to_drop = 0
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')

    for i in range(num_features_to_drop):
        index_of_attr_to_delete = -1
        lowest_score = 100
        for j in range(len(X[0])):
            X_copy = X.copy()
            new_x = np.delete(X_copy, i, 1)
            X1, X2, y1, y2 = train_test_split(new_x, y, test_size=0.30)
            clf.fit(X1, y1.flatten())
            score = clf.score(X2, y2.flatten())
            if score < lowest_score:
                lowest_score = score
                index_of_attr_to_delete = j
        X = np.delete(X, index_of_attr_to_delete, 1)

    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.30)
    clf.fit(X1, y1.flatten())
    score = clf.score(X2, y2.flatten())
    return score

def run_credit_approval():
    # hyper-parameters
    weight_type = 'inverse_distance'  # inverse_distance or distance   # make boolean
    k_value = 3
    normalization = False
    dist_metric = True

    # Get the file and Parse the data
    file_train = "credit_approval.arff"
    mat_train = Arff(file_train, label_count=1)
    data_train = mat_train.data[:, 0:-1]
    labels_train = mat_train.data[:, -1].reshape(-1, 1)
    column_types = mat_train.attr_types

    ranges_train = []
    if normalization:
        data_train, ranges_train = normalize(data_train, column_types)

    X1, X2, y1, y2 = train_test_split(data_train, labels_train, test_size=0.10)

    PClass = KNNClassifier(columntype=column_types, weight_type=weight_type, k_value=k_value, dist_metric=dist_metric,
                           ranges=ranges_train)
    PClass.fit(X1, y1)
    score = PClass.score(X2, y2)
    print(score)

def normalize(X, column_types):
    num_attributes = len(X[0])
    num_instances = len(X)
    col_max = np.nanmax(X, axis=0)
    col_min = np.nanmin(X, axis=0)
    ranges = col_max - col_min

    for i in range(num_attributes):
        if column_types[i] == 'nominal':
            continue
        for j in range(num_instances):
            X[j][i] = (X[j][i] - col_min[i]) / (col_max[i] - col_min[i])
    return X, ranges



def start():
    # Files to be read
    arff_files = [
        "credit_approval",      # 0
        "diabetes",             # 1
        "diabetes_test",        # 2
        "housing_test",         # 3
        "housing_train",        # 4
        "seismic-bumps_test",   # 5
        "seismic-bumps_train",  # 6
        "telescope_test",       # 7
        "telescope_train",      # 8
        "ionosphere"           # 9
    ]

    # Which type to run
    run_normal = False
    run_sk_learn = False
    run_sk_with_PCA_and_wrapper = True

    # hyper-parameters
    test_index = 9
    train_index = 9
    weight_type = 'inverse_distance'   # inverse_distance or distance
    k_value = 3
    normalization = False
    dist_metric = False

    # Get the file and Parse the data
    file_test = arff_files[test_index] + ".arff"
    file_train = arff_files[train_index] + ".arff"
    print("file_test: ", file_test, "file_train: ", file_train)
    mat_test = Arff(file_test, label_count=1)
    mat_train = Arff(file_train, label_count=1)
    data_test = mat_test.data[:,0:-1]
    labels_test = mat_test.data[:,-1].reshape(-1,1)
    data_train = mat_train.data[:,0:-1]
    labels_train = mat_train.data[:,-1].reshape(-1,1)
    column_types = mat_test.attr_types

    ranges_train = []
    if normalization:
        data_test, ranges_test = normalize(data_test, column_types)
        data_train, ranges_train = normalize(data_train, column_types)

    # Run the different types
    if run_normal:
        PClass = KNNClassifier(columntype=column_types, weight_type=weight_type, k_value=k_value, dist_metric=dist_metric, ranges=ranges_train)
        PClass.fit(data_train, labels_train)
        score = PClass.score(data_test, labels_test)
        print(score)

    if run_sk_learn:
        clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
        # clf = KNeighborsRegressor(n_neighbors=3)
        clf.fit(data_train, labels_train.flatten())
        score = clf.score(data_test, labels_test.flatten())
        print(score)

    if run_sk_with_PCA_and_wrapper:
        pca = PCA(n_components=33)  # 34 attributes total
        result_data = pca.fit_transform(data_train)
        score = wrapper(result_data, labels_train)
        print(score)

start()