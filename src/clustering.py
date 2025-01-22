import gin
import importlib

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.optimize import linear_sum_assignment

from collections import defaultdict


@gin.configurable(allowlist=None)
def clustering_algorithm(module: str, class_: str, **kwargs: dict):
    module = importlib.import_module(module)
    cls = getattr(module, class_)
    return cls(**kwargs)


class Clustering:
    def __init__(self, clustering_algorithm):
        self.clustering_algorithm = clustering_algorithm

    def compute_clusters(self, embeddings: np.ndarray):
        cluster_assignments = self.clustering_algorithm.fit_predict(embeddings)
        return cluster_assignments

    @staticmethod
    def cluster_formating(cluster_list: np.ndarray):
        """
        formatting
        [0, 1, 1, 2, 2, 2, 3, 3]
        to
        true_clusters = {
            1: [0],
            2: [1, 2],
            3: [3, 4, 5],
            4: [6, 7],
        }
        """
        clusters_formatted = defaultdict(list)
        for index, cluster_number in enumerate(cluster_list):
            clusters_formatted[cluster_number].append(index)
        return clusters_formatted

    @staticmethod
    def link_clusters(cluster_assignments: np.ndarray, labels: np.ndarray):

        labels_dict = defaultdict(set)
        cluster_dict = defaultdict(set)

        # add indices to sets for each label
        for i, val in enumerate(labels.tolist()):
            labels_dict[val].add(i)

        for i, val in enumerate(cluster_assignments.tolist()):
            cluster_dict[val].add(i)

        # calculate intersections and link clusters and events
        cluster_evt_dict = {}
        for cluster, cluster_set in cluster_dict.items():
            max_intersect = 0
            max_evt = None
            for evt, evt_set in labels_dict.items():
                intersect = len(cluster_set & evt_set)
                if intersect > max_intersect:
                    max_intersect = intersect
                    max_evt = evt
            cluster_evt_dict[cluster] = max_evt

        vfunc = np.vectorize(cluster_evt_dict.get)
        return vfunc(cluster_assignments)

    @staticmethod
    def link_cluster_hungarian_method(true_clusters: dict, predicted_clusters: dict):

        intersection_matrix = np.zeros((len(true_clusters), len(predicted_clusters)))

        for i, true_cluster in true_clusters.items():
            for j, predicted_cluster in predicted_clusters.items():
                intersection_matrix[i, j] = len(set(true_cluster) & set(predicted_cluster))

        cost_matrix = np.max(intersection_matrix) - intersection_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matching_result = {true_idx: pred_idx for true_idx, pred_idx in zip(row_ind, col_ind)}

        return matching_result

    @staticmethod
    def prepare_labels_for_metrics(true_clusters: dict, predicted_clusters: dict, matching_result: dict):

        true_labels = []
        pred_labels = []

        for true_cluster_id, predicted_cluster_id in matching_result.items():
            true_set = set(true_clusters[true_cluster_id])
            predicted_set = set(predicted_clusters[predicted_cluster_id])

            common_elements = true_set & predicted_set
            only_in_true = true_set - predicted_set
            only_in_predicted = predicted_set - true_set

            for _ in common_elements:
                true_labels.append(true_cluster_id)
                pred_labels.append(predicted_cluster_id)

            for _ in only_in_true:
                true_labels.append(true_cluster_id)
                pred_labels.append(-2)

            for _ in only_in_predicted:
                pred_labels.append(predicted_cluster_id)
                true_labels.append(-1)

        assert len(true_labels) == len(pred_labels), "The label lists are unbalanced!"

        matching_result_rev = {v:k for k,v in matching_result.items()}
        pred_labels = [matching_result_rev.get(p,-2) for p in pred_labels]
        return true_labels, pred_labels

    def cluster_and_link_fixed(self, embeddings: np.ndarray, labels: np.ndarray):
        cluster_assignments = self.compute_clusters(embeddings)
        return self.link_clusters(cluster_assignments, labels)

    def cluster_and_link(self, embeddings: np.ndarray, labels: np.ndarray):
        cluster_assignments = self.compute_clusters(embeddings)
        labels = self.cluster_formating(labels)
        cluster_assignments_formatted = self.cluster_formating(cluster_assignments)

        matching_result = self.link_cluster_hungarian_method(labels, cluster_assignments_formatted)
        labels, pred_labels = self.prepare_labels_for_metrics(labels, cluster_assignments_formatted, matching_result)

        return labels, pred_labels, cluster_assignments