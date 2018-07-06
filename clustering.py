import time

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from benchmark import Benchmark


class Clustering(Benchmark):
    start_time = None
    end_time = None

    # algorithm configurations
    method_name = 'Verse-PPR'
    dataset_name = 'Test-Data'
    performance_function = 'nmi'
    logistic_regression_model = None
    node_label_predictions = []
    n_clusters = 2
    random_seed = None

    # performance evaluation methods
    NMI = 'nmi'
    SILHOUETTE = 'silhouette'
    BOTH = 'both'

    # feature and label space
    embeddings = []
    node_labels = []

    # model and its prediction
    k_means = None

    def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='nmi',
                 embeddings=None, node_labels=None, n_clusters=2):
        """
        Initialize classification algorithm with customized configuration parameters
        :param method_name:
        :param dataset_name:
        :param performance_function:
        :param embeddings:
        :param node_labels:
        :param n_clusters:
        """
        
        print('Initialize clustering experiment with {} on {} evaluated through {}!'
              .format(method_name, dataset_name, performance_function))

        self.method_name = method_name
        self.dataset_name = dataset_name
        self.performance_function = performance_function
        self.embeddings = embeddings
        self.node_labels = node_labels
        self.n_clusters = n_clusters
        self.random_seed = random_seed

    def train(self):
        """
        Train clustering through k-means approach
        :return:
        """
        print('Train clustering experiment with {} on {} evaluated through {}!'
              .format(self.method_name, self.dataset_name, self.performance_function))

        self.start_time = time.time()

        self.k_means = KMeans(n_clusters=self.n_clusters, init='k-means++', n_jobs=-1,
                              n_init=1, random_state=self.random_seed)
        self.k_means.fit(self.embeddings)

        self.end_time = time.time()

        total_train_time = round(self.end_time - self.start_time, 2)
        print('Trained clustering experiment in {} sec.!'.format(total_train_time))

        return self.k_means

    def predict(self):
        """
        Predict clustering of each sample, based on nearest centroid
        :return:
        """
        print('Predict clustering experiment with {} on {} evaluated through {}!'
              .format(self.method_name, self.dataset_name, self.performance_function))

        self.start_time = time.time()

        self.node_label_predictions = self.k_means.predict(self.embeddings)

        self.end_time = time.time()

        total_prediction_time = round(self.end_time - self.start_time, 2)
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

        return results
