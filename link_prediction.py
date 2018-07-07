import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from typing import List
from tqdm import tqdm
import pickle

from verse.python.convert import map_nodes_to_ids

from benchmark import Benchmark


class LinkPrediction(Benchmark):
    """
    link prediction through logistic regression on edgewise features
    wrapper for customizable initialization, training, prediction and evaluation
    """

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

    # define train size
    train_size = 0.5

    def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='both', neg_edges=None,
                 node_embeddings=None, new_edges=None, vector_operator='hadamard', random_seed=None,
                 node2id_filepath=None, edge_list=None, ignore_new_nodes=True):
        """
        Initialize link prediction algorithm with customized configuration parameters
        Compute edgewise features
        :param method_name:
        :param dataset_name:
        :param performance_function:
        :param neg_edges: Edges which aren't in the delta of timesteps t_0 and t_1. We sample the edges, if this parameter isn't set. Identifier of nodes is the original id which is also used in node2id as key, would be the Q-id in wikidata.
        :param node_embeddings:
        :param new_edges: Delta of edges between timesteps t_0 and t_1: All edges which are new in t_1, doesn't consider edges which were removed. Identifier of nodes is the original id which is also used in node2id as key, would be the Q-id in wikidata.
        :param vector_operator:
        :param random_seed:
        :param node2id_filepath: Path where the node2id dict is located. If this argument is None, the node ids are calculated from the new_edges and neg_edges
        :param edge_list: List of all edges in graph at timestep t_0 with original id.
        :param ignore_new_nodes: Ignore edges with new nodes in new_edges and neg_edges
        """

        print('Initialize link prediction experiment with {} on {} evaluated through {} on {}% train data!'
              .format(method_name, dataset_name, performance_function, self.train_size * 100.00))

        self.method_name = method_name
        self.dataset_name = dataset_name
        self.performance_function = performance_function
        self.node_embeddings = node_embeddings
        self.random_seed = random_seed
        self.edge_embeddings = np.empty(shape=(0, np.shape(self.node_embeddings)[1]))
        self.logistic_regression_model = None
        self.edge_label_predictions = []
        self.edge_labels = []

        if node2id_filepath is not None:
            print("Use ids for nodes from node2id dict")
            with open(node2id_filepath, 'rb') as file:
                self.node2id = pickle.load(file)
            if not ignore_new_nodes:
                # Find all nodes
                nodes = set()
                for edge in tqdm(new_edges):
                    node1, node2 = edge
                    nodes.add(node1)
                    nodes.add(node2)

                if neg_edges is not None:
                    for edge in tqdm(neg_edges):
                        node1, node2 = edge
                        nodes.add(node1)
                        nodes.add(node2)
                # Assign an id to new nodes
                max_id = max(self.node2id.values())
                for node in nodes:
                    if node not in self.node2id.keys():
                        max_id += 1
                        self.node2id[node] = max_id
        else:
            print("Calculate ids for nodes")
            nodes = set()
            for edge in tqdm(edge_list):
                node1, node2 = edge
                nodes.add(node1)
                nodes.add(node2)
            if not ignore_new_nodes:
                for edge in tqdm(new_edges):
                    node1, node2 = edge
                    nodes.add(node1)
                    nodes.add(node2)
                if neg_edges is not None:
                    for edge in tqdm(neg_edges):
                        node1, node2 = edge
                        nodes.add(node1)
                        nodes.add(node2)
            _, self.node2id, _ = map_nodes_to_ids(nodes)
        new_edges_converted = []
        for edge in new_edges:
            node1, node2 = edge
            if ignore_new_nodes:
                # Filter edges with new nodes
                if node1 not in self.node2id.keys() or node2 not in self.node2id.keys():
                    continue
            new_edges_converted.append((self.node2id[node1], self.node2id[node2]))

        neg_edges_converted = []
        for edge in neg_edges:
            node1, node2 = edge
            if ignore_new_nodes:
                # Filter edges with new nodes
                if node1 not in self.node2id.keys() or node2 not in self.node2id.keys():
                    continue
            neg_edges_converted.append((self.node2id[node1], self.node2id[node2]))

        self.new_edges = new_edges_converted
        self.vector_operator = vector_operator
        self.neg_edges = neg_edges_converted

        assert len(self.edge_labels) == 0, str(len(self.edge_labels))
        assert len(self.edge_embeddings) == 0, str(len(self.edge_embeddings))

        print('Compute edgewise features based on {} operator!'.format(self.vector_operator))
        self.compute_edgewise_features(self.new_edges, 1)

        assert len(self.edge_labels) == len(self.new_edges), "{} {}".format(len(self.edge_labels), len(self.new_edges))
        assert len(self.edge_embeddings) == len(self.new_edges), "{} {}".format(len(self.edge_embeddings),
                                                                                len(self.new_edges))

        if self.neg_edges is None:
            self.compute_edgewise_features(self.sample_non_existing_edges(len(self.new_edges)), 0)
        else:
            self.compute_edgewise_features(self.neg_edges, 0)

        self.edge_embeddings_train, self.edge_embeddings_test, self.edge_labels_train, self.edge_labels_test = \
            train_test_split(self.edge_embeddings, self.edge_labels, train_size=self.train_size,
                             random_state=self.random_seed)

    def compute_edgewise_features(self, edges, label):
        """
        Compute new edge feature space based on configured vector operator
        :param edges:
        :param label:
        :return:
        """
        vectors = []
        for edge in tqdm(edges):
            n1 = np.array(self.node_embeddings[edge[0]])
            n2 = np.array(self.node_embeddings[edge[1]])

            if self.vector_operator == self.AVERAGE:
                vectors.append(self.average_op(n1, n2))
            elif self.vector_operator == self.CONCAT:
                vectors.append(self.concat_op(n1, n2))
            elif self.vector_operator == self.HADAMARD:
                vectors.append(self.hadamard_op(n1, n2))
            elif self.vector_operator == self.WEIGHTED_L1:
                vectors.append(self.weighted_l1_op(n1, n2))
            elif self.vector_operator == self.WEIGHTED_L2:
                vectors.append(self.average_op(n1, n2))
            else:
                raise NotImplementedError('This vector operation is not supported')

            self.edge_labels.append(label)
        self.edge_embeddings = np.concatenate((self.edge_embeddings, vectors), axis=0)

    # TODO: implement all vector operators used in VERSE experiments for calculating edgewise embeddings
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

    def train(self):
        """
        Train through logistic regression
        :return:
        """
        print('Train link prediction experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        start_time = time.time()

        self.logistic_regression_model = LogisticRegression(penalty='l2', C=1., solver='saga', multi_class='ovr',
                                                            verbose=1, class_weight='balanced',
                                                            random_state=self.random_seed, n_jobs=-1)
        self.logistic_regression_model.fit(self.edge_embeddings_train, self.edge_labels_train)

        end_time = time.time()

        total_train_time = round(end_time - start_time, 2)
        print('Trained link prediction experiment in {} sec.!'.format(total_train_time))

        return self.logistic_regression_model

    def predict(self):
        """
        Predict class of each sample, based on pre-trained model
        :return:
        """
        print('Predict link prediction experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        start_time = time.time()

        self.edge_label_predictions = self.logistic_regression_model.predict(self.edge_embeddings_test)

        end_time = time.time()

        total_prediction_time = round(end_time - start_time, 2)
        print('Predicted link prediction experiment in {} sec.!'.format(total_prediction_time))

        return self.edge_label_predictions

    def evaluate(self):
        """
        Evaluate prediction results through already pre-defined performance function(s), return results as a dict
        :return:
        """
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
        else:
            raise NotImplementedError('The evaluation metric {} is not supported'.format(self.performance_function))

        return results
