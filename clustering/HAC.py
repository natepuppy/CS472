import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from tools.arff import Arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.distances = []
        self.cluster_points = []
        self.cluster_indexes = []
        self.errors = []
        self.centroids = []

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        # calculate the distance between each point and put it in a 2D array
        # self.distances = [[float("inf") for x in range(len(X))] for y in range(len(X))]

        self.distances = [[0 for x in range(len(X))] for y in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X)):
                self.distances[i][j] = np.linalg.norm(X[i] - X[j])   # did have if i < j: here

        # create initial clusters (all individual)
        clusters = []
        for i in range(len(X)):
            clusters.append([i])

        min_k = self.k[np.argmin(self.k)]

        while len(clusters) > min_k:
            # find two points that are the closest
            cluster_1_index = float("inf")
            cluster_2_index = float("inf")
            min_distance = float("inf")
            for i in range(len(clusters)):
                for j in range(len(clusters)):
                    if i < j:
                        dist = self.find_distance(clusters[i], clusters[j])
                        if dist < min_distance:
                            min_distance = dist
                            cluster_1_index = i
                            cluster_2_index = j

            # put these points in the same cluster, or if they are already clusters, combine the clusters
            clusters[cluster_1_index] = clusters[cluster_1_index] + clusters[cluster_2_index]
            clusters.pop(cluster_2_index)
            if len(clusters) in self.k:
                cluster_points = [[] for y in range(len(clusters))]
                for i in range(len(clusters)):
                    for index in clusters[i]:
                        cluster_points[i].append(X[index])

                self.cluster_points = cluster_points
                self.cluster_indexes = clusters
                self.calculate_centroid(cluster_points)
                self.SSE()
                self.save_clusters("debug_hac_" + self.link_type + ".txt", len(clusters))


        return self

    # def final_calculations(self, X, clusters):
    #     cluster_points = [[] for y in range(len(clusters))]
    #     for i in range(len(clusters)):
    #         for index in clusters[i]:
    #             cluster_points[i].append(X[index])
    #
    #     self.cluster_points = cluster_points
    #     self.cluster_indexes = clusters
    #     self.calculate_centroid(cluster_points)
    #     self.SSE()
    #     self.save_clusters(filename)



    def find_distance(self, cluster1, cluster2):
        if self.link_type == 'single':
            smallest_distance = float("inf")
            for i in cluster1:
                for j in cluster2:
                    dist = self.distances[i][j]
                    if dist < smallest_distance:
                        smallest_distance = dist
            return smallest_distance
        else:
            largest_distance = 0.0
            for i in cluster1:
                for j in cluster2:
                    # HERE this could have an index j that is greater than i
                    dist = self.distances[i][j]
                    if dist > largest_distance:
                        largest_distance = dist
            return largest_distance

    def calculate_centroid(self, clusters):
        centroids = []
        for cluster in clusters:
            centroid = np.mean(cluster, axis=0)
            centroids.append(centroid)
        self.centroids = centroids

    def SSE(self):
        total_errors = []
        for i in range(len(self.cluster_points)):
            errors = []
            for instance in self.cluster_points[i]:   # each instance in a cluster
                dist = np.linalg.norm(self.centroids[i] - instance)
                errors.append(dist ** 2)
            total_errors.append(sum(errors))
        self.errors = total_errors

    def save_clusters(self,filename,k):
        f = open(filename,"w+")    # This is the file you are going to write to, not the data you are getting the data from
        f.write("{:d}\n".format(k))
        f.write("{:.4f}\n\n".format(sum(self.errors)))
        for i in range(len(self.centroids)):
            f.write(np.array2string(self.centroids[i],precision=4,separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.cluster_indexes[i])))
            f.write("{:.4f}\n\n".format(self.errors[i]))
        f.close()

def normalize(X):
    num_attributes = len(X[0])
    num_instances = len(X)
    col_max = np.nanmax(X, axis=0)
    col_min = np.nanmin(X, axis=0)

    for i in range(num_attributes):
        for j in range(num_instances):
            X[j][i] = (X[j][i] - col_min[i]) / (col_max[i] - col_min[i])
    return X

def start():
    # Files to be read
    arff_files = [
        "abalone",              # 0
        "seismic-bumps_train",  # 1
        "iris"                  # 2
    ]

    # Hyper-Parameters
    index = 0
    label_count = 0        # 0 includes the output, 1 excludes the output
    normalize_data = True
    k = [5]
    link_type = 'complete'  # 'single' or 'complete'
    sk_learn = False

    # Get the file and Parse the data
    file = arff_files[index] + ".arff"
    mat = Arff(file, label_count=label_count)

    if label_count != 0:
        data = mat.data[:, 0:-label_count]
    else:
        data = mat.data

    if normalize_data:
        data = normalize(data)

    if sk_learn:
        clustering = AgglomerativeClustering(n_clusters=k[0]).fit(data)
        print()
    else:
        HACClustering(k=k, link_type=link_type).fit(data)



start()


# TODO: pass in a range of k values instead of just one
# TODO: graph the SSE (for each value of k)
# TODO: Implement sk_learn


# scaler = MinMaxScaler()
# scaler.fit(data)
# data = scaler.transform(data)











# if index == 28:
#     print()

# self.save_clusters(f, index, cluster_1_index, cluster_2_index, min_distance, clusters)


# def save_clusters(self, f, index, cluster_1_index, cluster_2_index, min_distance, cluster_indexes):
#     # filename = "debug_hac_single.txt"
#     # f = open(filename, "w+")
#
#     f.write("***************\nIteration " + str(index) + "\n***************")
#     f.write("\nMerging clusters " + str(cluster_1_index) + " and " + str(cluster_2_index) + "\tDistance  ")
#     f.write("{:.6f}\n\n".format(min_distance))
#     for i in range(len(cluster_indexes)):
#         my_string = ' '.join(map(str, cluster_indexes[i]))
#         f.write("Cluster " + str(i) + ": " + my_string + "\n")
#     f.write("\n\n")


# def find_distance(self, cluster1, cluster2):
#     if self.link_type == 'single':
#         smallest_distance = float("inf")
#         for i in cluster1:
#             for j in cluster2:
#                 if i < j:
#                     dist = self.distances[i][j]
#                     if dist < smallest_distance:
#                         smallest_distance = dist
#         return smallest_distance
#     else:
#         largest_distance = 0
#         for i in cluster1:
#             for j in cluster2:
#                 if i < j:
#                     dist = self.distances[i][j]
#                     if dist > largest_distance and dist != float("inf"):
#                         largest_distance = dist
#         return largest_distance


# import numpy as np
# from sklearn.base import BaseEstimator, ClusterMixin
# from tools.arff import Arff
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.cluster import AgglomerativeClustering
#
#
# class HACClustering(BaseEstimator, ClusterMixin):
#
#     def __init__(self, k=3, link_type='single'):  ## add parameters here
#         """
#         Args:
#             k = how many final clusters to have
#             link_type = single or complete. when combining two clusters use complete link or single link
#         """
#         self.link_type = link_type
#         self.k = k
#         self.distances = []
#         self.cluster_points = []
#         self.cluster_indexes = []
#         self.errors = []
#         self.centroids = []
#
#     def fit(self, X, y=None):
#         """ Fit the data; In this lab this will make the K clusters :D
#         Args:
#             X (array-like): A 2D numpy array with the training data
#             y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
#         Returns:
#             self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
#         """
#         # calculate the distance between each point and put it in a 2D array
#         # self.distances = [[float("inf") for x in range(len(X))] for y in range(len(X))]
#
#         self.distances = [[0 for x in range(len(X))] for y in range(len(X))]
#         for i in range(len(X)):
#             for j in range(len(X)):
#                 self.distances[i][j] = np.linalg.norm(X[i] - X[j])  # did have if i < j: here
#
#         # create initial clusters (all individual)
#         clusters = []
#         for i in range(len(X)):
#             clusters.append([i])
#
#         min_index = np.argmin(self.k)
#
#         while len(clusters <):
#             # find two points that are the closest
#             cluster_1_index = float("inf")
#             cluster_2_index = float("inf")
#             min_distance = float("inf")
#             for i in range(len(clusters)):
#                 for j in range(len(clusters)):
#                     if i < j:
#                         dist = self.find_distance(clusters[i], clusters[j])
#                         if dist < min_distance:
#                             min_distance = dist
#                             cluster_1_index = i
#                             cluster_2_index = j
#
#             # put these points in the same cluster, or if they are already clusters, combine the clusters
#             clusters[cluster_1_index] = clusters[cluster_1_index] + clusters[cluster_2_index]
#             clusters.pop(cluster_2_index)
#             if len(clusters) == self.k:
#                 cluster_points = [[] for y in range(len(clusters))]
#                 for i in range(len(clusters)):
#                     for index in clusters[i]:
#                         cluster_points[i].append(X[index])
#
#                 self.cluster_points = cluster_points
#                 self.cluster_indexes = clusters
#                 self.calculate_centroid(cluster_points)
#                 self.SSE()
#                 self.save_clusters("debug_hac_" + self.link_type + ".txt")
#
#         return self
#
#     # def final_calculations(self, X, clusters):
#     #     cluster_points = [[] for y in range(len(clusters))]
#     #     for i in range(len(clusters)):
#     #         for index in clusters[i]:
#     #             cluster_points[i].append(X[index])
#     #
#     #     self.cluster_points = cluster_points
#     #     self.cluster_indexes = clusters
#     #     self.calculate_centroid(cluster_points)
#     #     self.SSE()
#     #     self.save_clusters(filename)
#
#     def find_distance(self, cluster1, cluster2):
#         if self.link_type == 'single':
#             smallest_distance = float("inf")
#             for i in cluster1:
#                 for j in cluster2:
#                     dist = self.distances[i][j]
#                     if dist < smallest_distance:
#                         smallest_distance = dist
#             return smallest_distance
#         else:
#             largest_distance = 0.0
#             for i in cluster1:
#                 for j in cluster2:
#                     # HERE this could have an index j that is greater than i
#                     dist = self.distances[i][j]
#                     if dist > largest_distance:
#                         largest_distance = dist
#             return largest_distance
#
#     def calculate_centroid(self, clusters):
#         centroids = []
#         for cluster in clusters:
#             centroid = np.mean(cluster, axis=0)
#             centroids.append(centroid)
#         self.centroids = centroids
#
#     def SSE(self):
#         total_errors = []
#         for i in range(len(self.cluster_points)):
#             errors = []
#             for instance in self.cluster_points[i]:  # each instance in a cluster
#                 dist = np.linalg.norm(self.centroids[i] - instance)
#                 errors.append(dist ** 2)
#             total_errors.append(sum(errors))
#         self.errors = total_errors
#
#     def save_clusters(self, filename):
#         f = open(filename,
#                  "w+")  # This is the file you are going to write to, not the data you are getting the data from
#         f.write("{:d}\n".format(self.k))
#         f.write("{:.4f}\n\n".format(sum(self.errors)))
#         for i in range(len(self.centroids)):
#             f.write(np.array2string(self.centroids[i], precision=4, separator=","))
#             f.write("\n")
#             f.write("{:d}\n".format(len(self.cluster_indexes[i])))
#             f.write("{:.4f}\n\n".format(self.errors[i]))
#         f.close()
#
#
# def normalize(X):
#     num_attributes = len(X[0])
#     num_instances = len(X)
#     col_max = np.nanmax(X, axis=0)
#     col_min = np.nanmin(X, axis=0)
#
#     for i in range(num_attributes):
#         for j in range(num_instances):
#             X[j][i] = (X[j][i] - col_min[i]) / (col_max[i] - col_min[i])
#     return X
#
#
# def start():
#     # Files to be read
#     arff_files = [
#         "abalone",  # 0
#         "seismic-bumps_train",  # 1
#         "iris"  # 2
#     ]
#
#     # Hyper-Parameters
#     index = 0
#     label_count = 0  # 0 includes the output, 1 excludes the output
#     normalize_data = True
#     k = 5
#     link_type = 'complete'  # 'single' or 'complete'
#     sk_learn = False
#
#     # Get the file and Parse the data
#     file = arff_files[index] + ".arff"
#     mat = Arff(file, label_count=label_count)
#
#     if label_count != 0:
#         data = mat.data[:, 0:-label_count]
#     else:
#         data = mat.data
#
#     if normalize_data:
#         data = normalize(data)
#
#     if sk_learn:
#         clustering = AgglomerativeClustering(n_clusters=k).fit(data)
#         print()
#     else:
#         HACClustering(k=k, link_type=link_type).fit(data).save_clusters("debug_hac_" + link_type + ".txt")
#
#
# start()
#
# # TODO: pass in a range of k values instead of just one
# # TODO: graph the SSE   (for each value of k)
# # TODO: Implement sk_learn
#
#
# # scaler = MinMaxScaler()
# # scaler.fit(data)
# # data = scaler.transform(data)
#
#
# # if index == 28:
# #     print()
#
# # self.save_clusters(f, index, cluster_1_index, cluster_2_index, min_distance, clusters)
#
#
# # def save_clusters(self, f, index, cluster_1_index, cluster_2_index, min_distance, cluster_indexes):
# #     # filename = "debug_hac_single.txt"
# #     # f = open(filename, "w+")
# #
# #     f.write("***************\nIteration " + str(index) + "\n***************")
# #     f.write("\nMerging clusters " + str(cluster_1_index) + " and " + str(cluster_2_index) + "\tDistance  ")
# #     f.write("{:.6f}\n\n".format(min_distance))
# #     for i in range(len(cluster_indexes)):
# #         my_string = ' '.join(map(str, cluster_indexes[i]))
# #         f.write("Cluster " + str(i) + ": " + my_string + "\n")
# #     f.write("\n\n")
#
#
# # def find_distance(self, cluster1, cluster2):
# #     if self.link_type == 'single':
# #         smallest_distance = float("inf")
# #         for i in cluster1:
# #             for j in cluster2:
# #                 if i < j:
# #                     dist = self.distances[i][j]
# #                     if dist < smallest_distance:
# #                         smallest_distance = dist
# #         return smallest_distance
# #     else:
# #         largest_distance = 0
# #         for i in cluster1:
# #             for j in cluster2:
# #                 if i < j:
# #                     dist = self.distances[i][j]
# #                     if dist > largest_distance and dist != float("inf"):
# #                         largest_distance = dist
# #         return largest_distance
#

