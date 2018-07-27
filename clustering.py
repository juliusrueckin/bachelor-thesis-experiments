import time
import pickle
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from benchmark import Benchmark


class Clustering(Benchmark):

    # performance evaluation methods
    NMI = 'nmi'
    SILHOUETTE = 'silhouette'
    BOTH = 'both'

    def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='nmi',
                 embeddings=None, node_labels=None, node2id_filepath=None, n_clusters=2):
        """
        Initialize classification algorithm with customized configuration parameters
        :param method_name:
        :param dataset_name:
        :param performance_function:
        :param embeddings:
        :param node_labels:
        :param n_clusters:
        :param node2id_filepath:
        """
        
        print('Initialize clustering experiment with {} on {} evaluated through {}!'
              .format(method_name, dataset_name, performance_function))

        self.method_name = method_name
        self.dataset_name = dataset_name
        self.performance_function = performance_function
        self.embeddings = embeddings
        self.node_labels = node_labels
        self.n_clusters = n_clusters
        self.k_means = None
        self.node_label_predictions = []
        self.node2id_filepath = node2id_filepath

    def preprocess_data(self, random_seed=None):
        if self.node2id_filepath is not None:
            self.convert_node_labels()
        return self.embeddings, self.node_labels, self.embeddings, self.node_labels

    def convert_node_labels(self):
        node_labels_arr = np.zeros(np.shape(self.embeddings)[0], dtype=np.int32)
        node_to_id = {}
        with open(self.node2id_filepath, 'rb') as node2id_file:
            node_to_id = pickle.load(node2id_file)

        for node, index in node_to_id.items():
            node_labels_arr[index] = self.node_labels[node]

        self.node_labels = node_labels_arr

    def train(self, random_seed=None):
        """
        Train clustering through k-means approach
        :return:
        """
        print('Train clustering experiment with {} on {} evaluated through {}!'
              .format(self.method_name, self.dataset_name, self.performance_function))

        start_time = time.time()

        self.k_means = KMeans(n_clusters=self.n_clusters, init='k-means++', n_jobs=-1,
                              n_init=1, random_state=random_seed)
        self.k_means.fit(self.embeddings)

        end_time = time.time()

        total_train_time = round(end_time - start_time, 2)
        print('Trained clustering experiment in {} sec.!'.format(total_train_time))

        return self.k_means

    def predict(self):
        """
        Predict clustering of each sample, based on nearest centroid
        :return:
        """
        print('Predict clustering experiment with {} on {} evaluated through {}!'
              .format(self.method_name, self.dataset_name, self.performance_function))

        start_time = time.time()

        self.node_label_predictions = self.k_means.predict(self.embeddings)

        end_time = time.time()

        total_prediction_time = round(end_time - start_time, 2)
        print('Predicted clustering experiment in {} sec.!'.format(total_prediction_time))

        return self.node_label_predictions

    def evaluate(self):
        """
        Evaluate clustering quality through already pre-defined performance function(s), return results as a dict
        :return:
        """
        print('Evaluate clustering experiment with {} on {} evaluated through {}!'
              .format(self.method_name, self.dataset_name, self.performance_function))

        results = {}

        if self.performance_function == self.BOTH:
            results['nmi'] = float(normalized_mutual_info_score(self.node_labels, self.node_label_predictions))
            results['silhouette'] = float(silhouette_score(self.embeddings, self.node_label_predictions,
                                                           metric='cosine'))
        elif self.performance_function == self.NMI:
            results['nmi'] = float(normalized_mutual_info_score(self.node_labels, self.node_label_predictions))
        elif self.performance_function == self.SILHOUETTE:
            results['silhouette'] = float(silhouette_score(self.embeddings, self.node_label_predictions,
                                                           metric='cosine'))
        else:
            raise NotImplementedError('The evaluation metric {} is not supported'.format(self.performance_function))

        return results
