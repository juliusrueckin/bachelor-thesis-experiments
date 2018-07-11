import time
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from benchmark import Benchmark


class MultiLabelClassification(Benchmark):
    """
    Multi-label classification through k-nearest neighbor classifier or one vs. rest logistic regression
    Wrapper for customizable initialization, training, prediction and evaluation
    """

    # performance evaluation methods
    MACRO_F1 = 'macro_f1'
    MICRO_F1 = 'micro_f1'
    BOTH = 'both'

    # classifiers
    NEAREST_NEIGHBOR = 'nearest_neighbor'
    LOGISTIC_REGRESSION = 'logistic_regression'

    def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='both',
                 train_size=0.3, embeddings=None, node_labels=None, n_neighbors=5,
                 classifier='logistic_regression', node2id_filepath=None):
        """
        Initialize classification algorithm with customized configuration parameters
        Produce random train-test split
        :param method_name:
        :param dataset_name:
        :param performance_function:
        :param train_size:
        :param embeddings:
        :param node_labels:
        :param n_neighbors:
        :param classifier:
        :param node2id_filepath:
        """
        
        print('Initialize multi-label classification experiment with {} on {} evaluated through {} on {}% train data!'
              .format(method_name, dataset_name, performance_function, train_size * 100.00))

        self.method_name = method_name
        self.dataset_name = dataset_name
        self.performance_function = performance_function
        self.train_size = train_size
        self.embeddings = embeddings
        self.node_labels = node_labels
        self.n_neighbors = n_neighbors
        self.chosen_classifier = classifier
        self.node_label_predictions = []
        self.multi_label_model = None
        self.embeddings_train = []
        self.embeddings_test = []
        self.node_labels_train = []
        self.node_labels_test = []
        self.train_size = train_size
        self.node2id_filepath = node2id_filepath

    def preprocess_data(self, random_seed=None):
        self.convert_node_labels()
        self.embeddings_train, self.embeddings_test, self.node_labels_train, self.node_labels_test = \
            train_test_split(self.embeddings, self.node_labels, train_size=self.train_size,
                             test_size=1 - self.train_size, random_state=random_seed)

        return self.embeddings_train, self.embeddings_test, self.node_labels_train, self.node_labels_test

    def convert_node_labels(self):
        node_labels_binarized = MultiLabelBinarizer().fit_transform(self.node_labels.values())
        self.node_labels = dict(zip(self.node_labels.keys(), node_labels_binarized))

        node_labels_arr = np.zeros(np.shape(self.embeddings)[0] * np.shape(node_labels_binarized)[1])\
            .reshape(np.shape(self.embeddings)[0], np.shape(node_labels_binarized)[1])
        node_to_id = {}
        with open(self.node2id_filepath, 'rb') as node2id_file:
            node_to_id = pickle.load(node2id_file)

        for node, index in node_to_id.items():
            node_labels_arr[index] = self.node_labels[node]

        self.node_labels = node_labels_arr

    def train(self, random_seed=None):
        """
        Train through logistic regression
        :return:
        """
        print('Train multi-label classification experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        start_time = time.time()

        if self.chosen_classifier == self.NEAREST_NEIGHBOR:
            self.multi_label_model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance',
                                                          algorithm='auto', metric='cosine', n_jobs=-1)
        elif self.chosen_classifier == self.LOGISTIC_REGRESSION:
            self.multi_label_model = \
                OneVsRestClassifier(
                    LogisticRegression(penalty='l2', C=1., multi_class='ovr', solver='saga',verbose=1,
                                       class_weight='balanced', random_state=random_seed, n_jobs=-1), n_jobs=-1)

        self.multi_label_model.fit(self.embeddings_train, self.node_labels_train)

        end_time = time.time()

        total_train_time = round(end_time - start_time, 2)
        print('Trained multi-label classification experiment in {} sec.!'.format(total_train_time))

        return self.multi_label_model

    def predict(self):
        """
        Predict class of each sample, based on pre-trained model
        :return:
        """
        print('Predict multi-label classification experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        start_time = time.time()

        self.node_label_predictions = self.multi_label_model.predict(self.embeddings_test)

        end_time = time.time()

        total_prediction_time = round(end_time - start_time, 2)
        print('Predicted multi-label classification experiment in {} sec.!'.format(total_prediction_time))

        return self.node_label_predictions

    def evaluate(self):
        """
        Evaluate prediction results through already pre-defined performance function(s), return results as a dict
        :return:
        """
        print('Evaluate multi-label classification experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        results = {}

        if self.performance_function == self.BOTH:
            results['macro'] = float(f1_score(self.node_labels_test, self.node_label_predictions, average='macro'))
            results['micro'] = float(f1_score(self.node_labels_test, self.node_label_predictions, average='micro'))
        elif self.performance_function == self.MACRO_F1:
            results['macro'] = float(f1_score(self.node_labels_test, self.node_label_predictions, average='macro'))
        elif self.performance_function == self.MICRO_F1:
            results['micro'] = float(f1_score(self.node_labels_test, self.node_label_predictions, average='micro'))
        else:
            raise NotImplementedError('The evaluation metric {} is not supported'.format(self.performance_function))

        return results
