import math
import sys
from sklearn.metrics import silhouette_score
import numpy as np
from numpy import random
import mysymnmfsp

np.random.seed(0)

def avg_matrix(M):
    cols = len(M[0])
    rows = len(M)
    sum = 0
    for i in range(rows):
        for j in range(cols):
            sum += M[i][j]
    return (sum / (cols * rows))

def calc_distance(vector1, vector2):
    i = 0
    summ = 0
    for i in range(len(vector1)):
        summ += pow((vector1[i] - vector2[i]), 2)
    return math.sqrt(summ)

def adjust_vectors_per_iteration(cluster, vector, array_of_centroids, index_table):
    """
    Adjusts the vectors in the cluster based on the closest centroid.

    Args:
        cluster: A list of lists, where each sublist contains the vectors in a cluster.
        vector: A list of floats, representing a vector.
        array_of_centroids: A list of lists, where each sublist contains the coordinates of a centroid.
        index_table: A list of integers, where each integer represents the index of the cluster that a vector belongs to.

    Returns:
        None.
    """
    for i in range(len(vector)):
        dis_1 = calc_distance(vector[i], array_of_centroids[0])
        closest_cent = 0
        for j in range(1, len(array_of_centroids)):
            dis_2 = calc_distance(vector[i], array_of_centroids[j])
            if dis_2 < dis_1:
                dis_1 = dis_2
                closest_cent = j
        cluster[closest_cent].append(vector[i])
        index_table[i] = closest_cent


def assign_vectors_for_index_table(array_of_vectors, array_of_centroids, k, index_table):
    """
    Assigns vectors to clusters based on the closest centroid.

    Args:
        array_of_vectors: A list of lists, where each sublist contains the coordinates of a vector.
        array_of_centroids: A list of lists, where each sublist contains the coordinates of a centroid.
        k: The number of clusters.
        index_table: A list of integers, where each integer represents the index of the cluster that a vector belongs to.

    Returns:
        A list of lists, where each sublist contains the vectors in a cluster.
    """
    clusters_table = [[] for i in range(k)]
    cnt = 0
    for i in array_of_vectors:
        dis_1 = calc_distance(i, array_of_centroids[0])
        closest_centroid = 0
        for j in range(1, len(array_of_centroids)):
            dis_2 = calc_distance(i, array_of_centroids[j])
            if dis_2 < dis_1:
                dis_1 = dis_2
                closest_centroid = j
        clusters_table[closest_centroid].append(i)
        index_table[cnt] = closest_centroid
        cnt += 1
    return clusters_table


def calc_new_cent(clusters_table):
    """
    Calculates the new centroids for the clusters.

        Args:
            clusters_table: A list of lists, where each sublist contains the vectors in a cluster.

        Returns:
            A list of lists, where each sublist contains the coordinates of a new centroid.
        """

    new_centroid = [0 for i in range(len(clusters_table[0]))]
    for i in clusters_table:
        for j in range(len(clusters_table[0])):
            new_centroid[j] += i[j]
    for p in range(len(new_centroid)):
        new_centroid[p] = ((new_centroid[p]) / len(clusters_table))
    return new_centroid


def update_array_of_centroids(clusters_table, array_of_centroids, k):
    """
    Updates the array of centroids based on the current clusters table.

    Args:
        clusters_table: A list of lists, where each sublist represents a cluster.
        array_of_centroids: A list of centroids, where each centroid is a list of values.
        k: The number of clusters.

    Returns:
        A list of deltas, where each delta is a list of values representing the change in the centroid.
    """
    delta = [0 for i in range(k)]
    for i in range(k):
        new_cent = calc_new_cent(clusters_table[i])
        delta[i] = calc_distance(array_of_centroids[i], new_cent)
        array_of_centroids[i] = new_cent
    return delta


def check_table_of_centroid(array_of_centroids):
    """
    checks if the distance requirement if fulfilled
    """
    for i in array_of_centroids:
        if i >= 0.0001:
            return False
    return True


def Kmeans(K, iter, input_data):
    vectors = input_data
    index_table = [0 for i in range(len(vectors))]
    cnt = 0
    k = int(K)
    clusters_table = [[] for i in range(k)]
    array_of_centroids = []
    for i in range(k):
        array_of_centroids.append(vectors[i])
    adjust_vectors_per_iteration(clusters_table, vectors, array_of_centroids, index_table)
    deltas = update_array_of_centroids(clusters_table, array_of_centroids, k)
    check = check_table_of_centroid(deltas)
    if check:
        return array_of_centroids, index_table, vectors
    while cnt < iter:
        clusters_table = assign_vectors_for_index_table(vectors, array_of_centroids, k, index_table)
        deltas = update_array_of_centroids(clusters_table, array_of_centroids, k)
        check = check_table_of_centroid(deltas)
        if check:
            return array_of_centroids, index_table, vectors
        cnt += 1
    return array_of_centroids, index_table, vectors


# main function
def main():
    lst_of_arguments = sys.argv
    # Check the number of command-line arguments
    if (len(lst_of_arguments) != 3):
        print("An error has occurred")
        return 0
    first_argument = lst_of_arguments[1]
    # Check if the first argument is a valid number of clusters
    if first_argument.isdigit():
        first_argument = int(first_argument)
    else:
        print("An error has occurred")
        return 0
    file = lst_of_arguments[2]
    if (file[-4:] == ".txt"):
        try:
            data = np.genfromtxt(file, dtype=float, encoding=None, delimiter=",")
        except:
            print("An Error Has Occurred")
            return 1
    else:
        print("An Error Has Occurred")
        return 1
    list = data.tolist()
    #Check if the third argument exists and validate it
    n = len(list)
    k = first_argument
    H = [[0.0 for i in range(k)] for j in range(n)]
    #   Initializing H   #
    ###########################################
    W = mysymnmfsp.fit(list, H, 4, k)
    m = avg_matrix(W)
    max_val_interval = math.sqrt(m / k) * 2
    H_min = [0 for i in range(k)]
    H_max = [max_val_interval for i in range(k)]
    H = np.random.uniform(low=H_min, high=H_max, size=(n, k))
    H = H.tolist()
    finalH = mysymnmfsp.fit(list, H, 1, k)
    cluster_array_symnmf = np.argmax(finalH, axis=1)
    kmeans_table, cluster_array_kmeans,vectors = Kmeans(k, 300, list)
    symnmf_score = silhouette_score(list, cluster_array_symnmf)
    kmeans_score = silhouette_score(list, cluster_array_kmeans)
    formatted_string_symnmf = "{:.4f}".format(symnmf_score)
    formatted_string_kmeans = "{:.4f}".format(kmeans_score)
    print("nmf:", formatted_string_symnmf)
    print("kmeans:", formatted_string_kmeans)


if __name__ == "__main__":
    main()
