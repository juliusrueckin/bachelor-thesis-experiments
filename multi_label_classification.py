import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from benchmark import Benchmark


# multi-label classification through k-nearest neighbor classifier or one vs. rest logistic regression
# wrapper for customizable initialization, training, prediction and evaluation
class MultiLabelClassification(Benchmark):
    start_time = None
    end_time = None

    # algorithm configurations
    method_name = 'Verse-PPR'
    dataset_name = 'Test-Data'
    performance_function = 'Macro-F1'
    train_size = 0.3
    n_neighbors = 5
    chosen_classifier = ''
    random_seed = None

    # performance evaluation methods
    MACRO_F1 = 'macro_f1'
    MICRO_F1 = 'micro_f1'
    BOTH = 'both'

    # classifiers
    NEAREST_NEIGHBOR = 'nearest_neighbor'
    LOGISTIC_REGRESSION = 'logistic_regression'

    # feature and label space, where each sample has a set of classes
    embeddings = []
    node_labels = []

    # train-test split of feature and label space
    embeddings_train = []
    node_labels_train = []
    embeddings_test = []
    node_labels_test = []

    # model and its prediction
    multi_label_model = None
    node_label_predictions = []

    # initialize classification algorithm with customized configuration parameters
    # produce random train-test split
    def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='both',
                 train_size=0.3, embeddings=None, node_labels=None, n_neighbors=5, classifier='logistic_regression',
                 random_seed=None):
        print('Initialize multi-label classification experiment with {} on {} evaluated through {} on {}% train data!'
              .format(method_name, dataset_name, performance_function, train_size * 100.00))

        self.method_name = method_name
        self.dataset_name = dataset_name
        self.performance_function = performance_function
        self.train_size = train_size
        self.embeddings = embeddings
        self.node_labels = MultiLabelBinarizer().fit_transform(node_labels)
        self.n_neighbors = n_neighbors
        self.chosen_classifier = classifier
        self.random_seed = random_seed

        self.embeddings_train, self.embeddings_test, self.node_labels_train, self.node_labels_test = \
            train_test_split(self.embeddings, self.node_labels, train_size=train_size, test_size=1 - train_size,
                             random_state=self.random_seed)

    # train through k-nearest neighbor classifier
    def train(self):
        print('Train multi-label classification experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        self.start_time = time.time()

        if self.chosen_classifier == self.NEAREST_NEIGHBOR:
            self.multi_label_model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance',
                                                          algorithm='auto', metric='cosine', n_jobs=-1)
        elif self.chosen_classifier == self.LOGISTIC_REGRESSION:
            self.multi_label_model = \
                OneVsRestClassifier(
                    LogisticRegression(penalty='l2', C=1., multi_class='ovr', solver='saga',
                                       verbose=1, class_weight='balanced', random_state=self.random_seed), n_jobs=-1)

        self.multi_label_model.fit(self.embeddings_train, self.node_labels_train)

        self.end_time = time.time()

        total_train_time = round(self.end_time - self.start_time, 2)
        print('Trained multi-label classification experiment in {} sec.!'.format(total_train_time))

        return self.multi_label_model

    # predict class of each sample, based on pre-trained model
    def predict(self):
        print('Predict multi-label classification experiment with {} on {} evaluated through {} on {}% train data!'
              .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

        self.start_time = time.time()

        self.node_label_predictions = self.multi_label_model.predict(self.embeddings_test)

        self.end_time = time.time()

        total_prediction_time = round(self.end_time - self.start_time, 2)
        print('Predicted multi-label classification experiment in {} sec.!'.format(total_prediction_time))

        return self.node_label_predictions

    # evaluate prediction results through already pre-defined performance function(s), return results as a dict
    def evaluate(self):
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

        return results
