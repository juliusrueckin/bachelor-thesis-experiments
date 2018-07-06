import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from verse.python.convert import map_nodes_to_ids

from benchmark import Benchmark


# link prediction through logistic regression on edgewise features
# wrapper for customizable initialization, training, prediction and evaluation
class LinkPrediction(Benchmark):
    start_time = None
    end_time = None

    # algorithm configurations
    method_name = 'Verse-PPR'
    dataset_name = 'Test-Data'
    performance_function = 'Macro-F1'
    train_size = 0.5
    vector_operator = 'hadamard'
    random_seed = None

    # performance evaluation methods
    MACRO_F1 = 'macro_f1'
    MICRO_F1 = 'micro_f1'
    BOTH = 'both'

    # vector operators
    AVERAGE = 'avg'
    CONCAT = 'concat'
    HADAMARD = 'hadamard'
    WEIGHTED_L1 = 'weighted_l1'
    WEIGHTED_L2 = 'weighted_l2'

    # node and edge embedding feature space, new edges list and edge labels
    node_embeddings = []
    edge_embeddings = np.empty(shape=(0, 128))
    new_edges = []
    neg_edges = None
    edge_labels = []

    # train-test split of new edges feature space
    edge_embeddings_train = []
    edge_labels_train = []
    edge_embeddings_test = []
    edge_labels_test = []

    # model and its prediction
    logistic_regression_model = None
    edge_label_predictions = []

    # initialize link prediction algorithm with customized configuration parameters
    # compute edgewise features
    def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='both', neg_edges=None,
                 node_embeddings=None, new_edges=None, vector_operator='hadamard', random_seed=None):
        print('Initialize link prediction experiment with {} on {} evaluated through {} on {}% train data!'
              .format(method_name, dataset_name, performance_function, self.train_size * 100.00))

        self.method_name = method_name
        self.dataset_name = dataset_name
        self.performance_function = performance_function
        self.node_embeddings = node_embeddings
        self.random_seed = random_seed

        nodes = set()
        for edge in new_edges:
            node1, node2 = edge
            nodes.add(node1)
            nodes.add(node2)
        _, node2id, _ = map_nodes_to_ids(nodes)
        new_edges_converted = []
        for edge in new_edges:
            node1, node2 = edge
            new_edges_converted.append((node2id[node1], node2id[node2]))

        self.new_edges = new_edges_converted
        self.vector_operator = vector_operator
        self.neg_edges = neg_edges

        print('Compute edgewise features based on {} operator!'.format(self.vector_operator))
        self.compute_edgewise_features(self.new_edges, 1)
        if self.neg_edges is None:
            self.compute_edgewise_features(self.sample_non_existing_edges(len(self.new_edges)), 0)
        else:
            self.compute_edgewise_features(self.neg_edges, 0)

        self.edge_embeddings_train, self.edge_embeddings_test, self.edge_labels_train, self.edge_labels_test = \
            train_test_split(self.edge_embeddings, self.edge_labels, train_size=self.train_size,
                             random_state=self.random_seed)

    # compute new edge feature space based on configured vector operator
    def compute_edgewise_features(self, edges, label):
        for edge in edges:
            n1 = np.array(self.node_embeddings[edge[0]])
            n2 = np.array(self.node_embeddings[edge[1]])

            if self.vector_operator == self.AVERAGE:
                self.edge_embeddings = np.concatenate((self.edge_embeddings, [self.average_op(n1, n2)]), axis=0)
            elif self.vector_operator == self.CONCAT:
                self.edge_embeddings = np.concatenate((self.edge_embeddings, [self.concat_op(n1, n2)]), axis=0)
            elif self.vector_operator == self.HADAMARD:
                self.edge_embeddings = np.concatenate((self.edge_embeddings, [self.hadamard_op(n1, n2)]), axis=0)
            elif self.vector_operator == self.WEIGHTED_L1:
                self.edge_embeddings = np.concatenate((self.edge_embeddings, [self.weighted_l1_op(n1, n2)]), axis=0)
            elif self.vector_operator == self.WEIGHTED_L2:
                self.edge_embeddings = np.concatenate((self.edge_embeddings, [self.weighted_l2_op(n1, n2)]), axis=0)

            self.edge_labels.append(label)

    # implement all vector operators used in VERSE experiments for calculating edgewise embeddings
    @staticmethod
    def average_op(n1, n2):
        return (n1 + n2) / 2

    @staticmethod
    def concat_op(n1, n2):
        return np.concatenate((n1, n2), axis=0)

    @staticmethod
    def hadamard_op(n1, n2):
        return n1 * n2

    @staticmethod
    def weighted_l1_op(n1, n2):
        return n1 - n2

    @staticmethod
    def weighted_l2_op(n1, n2):
        return np.square(n1 - n2)

    def sample_non_existing_edges(self, num_of_sampled_edges):
        non_existing_edges = []
        num_of_nodes = np.shape(self.node_embeddings)[0]

        for i in range(num_of_sampled_edges):
            n1 = np.random.randint(num_of_nodes)
            n2 = np.random.randint(num_of_nodes)
            non_existing_edges.append([n1, n2])

        return non_existing_edges

    # train through logistic regression
    def train(self):
        print('Train link prediction experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        self.start_time = time.time()

        self.logistic_regression_model = LogisticRegression(penalty='l2', C=1., solver='saga', multi_class='ovr',
                                                            verbose=1, class_weight='balanced',
                                                            random_state=self.random_seed)
        self.logistic_regression_model.fit(self.edge_embeddings_train, self.edge_labels_train)

        self.end_time = time.time()

        total_train_time = round(self.end_time - self.start_time, 2)
        print('Trained link prediction experiment in {} sec.!'.format(total_train_time))

        return self.logistic_regression_model

    # predict class of each sample, based on pre-trained model
    def predict(self):
        print('Predict multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        self.start_time = time.time()

        self.edge_label_predictions = self.logistic_regression_model.predict(self.edge_embeddings_test)

        self.end_time = time.time()

        total_prediction_time = round(self.end_time - self.start_time, 2)
        print('Predicted link prediction experiment in {} sec.!'.format(total_prediction_time))

        return self.edge_label_predictions

    # evaluate prediction results through already pre-defined performance function(s), return results as a dict
    def evaluate(self):
        print('Evaluate link prediction experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        results = {}

        if self.performance_function == self.BOTH:
            results['macro'] = float(f1_score(self.edge_labels_test, self.edge_label_predictions, average='macro'))
            results['micro'] = float(f1_score(self.edge_labels_test, self.edge_label_predictions, average='micro'))
        elif self.performance_function == self.MACRO_F1:
            results['macro'] = float(f1_score(self.edge_labels_test, self.edge_label_predictions, average='macro'))
        elif self.performance_function == self.MICRO_F1:
            results['micro'] = float(f1_score(self.edge_labels_test, self.edge_label_predictions, average='micro'))

        return results
