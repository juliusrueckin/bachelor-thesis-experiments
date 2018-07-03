import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


# link prediction through logistic regression on edgewise features
# wrapper for customizable initialization, training, prediction and evaluation
class LinkPrediction:

	start_time = None
	end_time = None

	# algorithm configurations
	method_name = 'Verse-PPR'
	dataset_name = 'Test-Data'
	performance_function = 'Macro-F1'
	train_size = 0.5
	vector_operator = 'hadamard'

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

	# node and edge embedding feature space, and new edges list
	node_embeddings = []
	edge_embeddings = []
	new_edges = []

	# train-test split of new edges feature space
	edge_embeddings_train = []
	edge_embeddings_test = []

	# initialize classification algorithm with customized configuration parameters
	# compute new edge feature space
	# randomly split new edges feature space in one half train and test set
	def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='both',
				 node_embeddings=None, new_edges=None, vector_operator='hadamard'):
		print(
			'Initialize link prediction experiment with {} on {} evaluated through {} on {}% train data!'
			.format(method_name, dataset_name, performance_function, self.train_size * 100.00))

		self.method_name = method_name
		self.dataset_name = dataset_name
		self.performance_function = performance_function
		self.node_embeddings = node_embeddings
		self.new_edges = new_edges
		self.vector_operator = vector_operator

		self.compute_new_edges_feature_space()
		self.new_edges_train_test_split()

	def new_edges_train_test_split(self):
		self.edge_embeddings_train = self.edge_embeddings[]
		self.edge_embeddings_test = []

	# compute new edge feature space based on configured vector operator
	def compute_new_edges_feature_space(self):
		for edge in self.new_edges:
			n1 = np.array(self.node_embeddings[edge[0]])
			n2 = np.array(self.node_embeddings[edge[1]])

			if self.vector_operator == self.AVERAGE:
				self.edge_embeddings.append(self.average_op(n1, n2))
			elif self.vector_operator == self.CONCAT:
				self.edge_embeddings.append(self.concat_op(n1, n2))
			elif self.vector_operator == self.HADAMARD:
				self.edge_embeddings.append(self.hadamard_op(n1, n2))
			elif self.vector_operator == self.WEIGHTED_L1:
				self.edge_embeddings.append(self.weighted_l1_op(n1, n2))
			elif self.vector_operator == self.WEIGHTED_L2:
				self.edge_embeddings.append(self.weighted_l2_op(n1, n2))

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

	# train through logistic regression
	def train(self):
		print('Train multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
			  .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

		self.start_time = time.time()

		self.logistic_regression_model = LogisticRegression(penalty='l2', C=1., multi_class='multinomial',
															solver='saga',
															verbose=1, class_weight='balanced')
		self.logistic_regression_model.fit(self.embeddings_train, self.node_labels_train)

		self.end_time = time.time()

		total_train_time = round(self.end_time - self.start_time, 2)
		print('Trained multi-class classification experiment in {} sec.!'.format(total_train_time))

		return self.logistic_regression_model

	# predict class of each sample, based on pre-trained model
	def predict(self):
		print('Predict multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
			  .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

		self.start_time = time.time()

		self.node_label_predictions = self.logistic_regression_model.predict(self.embeddings_test)

		self.end_time = time.time()

		total_prediction_time = round(self.end_time - self.start_time, 2)
		print('Predicted multi-class classification experiment in {} sec.!'.format(total_prediction_time))

		return self.node_label_predictions

	# evaluate prediction results through already pre-defined performance function(s), return results as a dict
	def evaluate(self):
		print('Evaluate multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
			  .format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

		results = {}

		if self.performance_function == self.BOTH:
			results['macro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='macro')
			results['micro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='micro')
		elif self.performance_function == self.MACRO_F1:
			results['macro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='macro')
		elif self.performance_function == self.MICRO_F1:
			results['micro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='micro')

		return results


# multi-class classification through logistic regression
# wrapper for customizable initialization, training, prediction and evaluation
class MultiClassClassification:

	start_time = None
	end_time = None

	# algorithm configurations
	method_name = 'Verse-PPR'
	dataset_name = 'Test-Data'
	performance_function = 'Macro-F1'
	train_size = 0.3

	# performance evaluation methods
	MACRO_F1 = 'macro_f1'
	MICRO_F1 = 'micro_f1'
	BOTH = 'both'

	# feature and label space
	embeddings = []
	node_labels = []

	# train-test split of feature and label space
	embeddings_train = []
	node_labels_train = []
	embeddings_test = []
	node_labels_test = []

	# model and its prediction
	logistic_regression_model = None
	node_label_predictions = []

	# initialize classification algorithm with customized configuration parameters
	# produce random train-test split
	def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='both',
				 train_size=0.3, embeddings=None, node_labels=None):
		print('Initialize multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
			.format(method_name, dataset_name, performance_function, train_size*100.00))

		self.method_name = method_name
		self.dataset_name = dataset_name
		self.performance_function = performance_function
		self.train_size = train_size
		self.embeddings = embeddings
		self.node_labels = node_labels

		self.embeddings_train, self.embeddings_test, self.node_labels_train, self.node_labels_test = \
			train_test_split(self.embeddings, self.node_labels, train_size=train_size, test_size=1-train_size)

	# train through logistic regression
	def train(self):
		print('Train multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
			.format(self.method_name, self.dataset_name, self.performance_function, self.train_size*100.00))

		self.start_time = time.time()

		self.logistic_regression_model = LogisticRegression(penalty='l2', C=1., multi_class='multinomial', solver='saga',
															verbose=1, class_weight='balanced')
		self.logistic_regression_model.fit(self.embeddings_train, self.node_labels_train)

		self.end_time = time.time()

		total_train_time = round(self.end_time - self.start_time, 2)
		print('Trained multi-class classification experiment in {} sec.!'.format(total_train_time))

		return self.logistic_regression_model

	# predict class of each sample, based on pre-trained model
	def predict(self):
		print('Predict multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
			.format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

		self.start_time = time.time()

		self.node_label_predictions = self.logistic_regression_model.predict(self.embeddings_test)

		self.end_time = time.time()

		total_prediction_time = round(self.end_time - self.start_time, 2)
		print('Predicted multi-class classification experiment in {} sec.!'.format(total_prediction_time))

		return self.node_label_predictions

	# evaluate prediction results through already pre-defined performance function(s), return results as a dict
	def evaluate(self):
		print('Evaluate multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
			.format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

		results = {}

		if self.performance_function == self.BOTH:
			results['macro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='macro')
			results['micro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='micro')
		elif self.performance_function == self.MACRO_F1:
			results['macro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='macro')
		elif self.performance_function == self.MICRO_F1:
			results['micro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='micro')

		return results


# multi-label classification through k-nearest neighbor classifier or one vs. rest logistic regression
# wrapper for customizable initialization, training, prediction and evaluation
class MultiLabelClassification:
	start_time = None
	end_time = None

	# algorithm configurations
	method_name = 'Verse-PPR'
	dataset_name = 'Test-Data'
	performance_function = 'Macro-F1'
	train_size = 0.3
	n_neighbors = 5
	chosen_classifier = ''

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
				 train_size=0.3, embeddings=None, node_labels=None, n_neighbors=5, classifier='logistic_regression'):
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

		self.embeddings_train, self.embeddings_test, self.node_labels_train, self.node_labels_test = \
			train_test_split(self.embeddings, self.node_labels, train_size=train_size, test_size=1 - train_size)

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
					LogisticRegression(penalty='l2', C=1., multi_class='multinomial', solver='saga',
									   verbose=1, class_weight='balanced'), n_jobs=-1)

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
			results['macro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='macro')
			results['micro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='micro')
		elif self.performance_function == self.MACRO_F1:
			results['macro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='macro')
		elif self.performance_function == self.MICRO_F1:
			results['micro'] = f1_score(self.node_labels_test, self.node_label_predictions, average='micro')

		return results



class Clustering:

	start_time = None
	end_time = None

	# algorithm configurations
	method_name = 'Verse-PPR'
	dataset_name = 'Test-Data'
	performance_function = 'nmi'
	logistic_regression_model = None
	node_label_predictions = []
	n_clusters = 2

	# performance evaluation methods
	NMI = 'nmi'
	SILHOUETTE = 'silhouette'
	BOTH = 'both'

	# feature and label space
	embeddings = []
	node_labels = []

	# model and its prediction
	k_means = None
	node_label_predictions = []

	# initialize classification algorithm with customized configuration parameters
	def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='nmi',
				 embeddings=None, node_labels=None, n_clusters=2):
		print('Initialize clustering experiment with {} on {} evaluated through {}!'
				.format(self.method_name, self.dataset_name, self.performance_function))

		self.method_name = method_name
		self.dataset_name = dataset_name
		self.performance_function = performance_function
		self.embeddings = embeddings
		self.node_labels = node_labels
		self.n_clusters = n_clusters

	# train clustering through k-means approach
	def train(self):
		print('Train clustering experiment with {} on {} evaluated through {}!'
			  .format(self.method_name, self.dataset_name, self.performance_function))

		self.start_time = time.time()

		self.k_means = KMeans(n_clusters=self.n_clusters, init='k-means++', n_jobs=-1, n_init=1)
		self.k_means.fit(self.embeddings)

		self.end_time = time.time()

		total_train_time = round(self.end_time - self.start_time, 2)
		print('Trained clustering experiment in {} sec.!'.format(total_train_time))

		return self.k_means

	# predict clustering of each sample, based on nearest centroid
	def predict(self):
		print('Predict clustering experiment with {} on {} evaluated through {}!'
			  .format(self.method_name, self.dataset_name, self.performance_function))

		self.start_time = time.time()

		self.node_label_predictions = self.k_means.predict(self.embeddings)

		self.end_time = time.time()

		total_prediction_time = round(self.end_time - self.start_time, 2)
		print('Predicted clustering experiment in {} sec.!'.format(total_prediction_time))

		return self.node_label_predictions

	# evaluate clustering quality through already pre-defined performance function(s), return results as a dict
	def evaluate(self):
		print('Evaluate clustering experiment with {} on {} evaluated through {}!'
				.format(self.method_name, self.dataset_name, self.performance_function))

		results = {}

		if self.performance_function == self.BOTH:
			results['nmi'] = normalized_mutual_info_score(self.node_labels, self.node_label_predictions)
			results['silhouette'] = silhouette_score(self.embeddings, self.node_label_predictions,
													 metric='cosine')
		elif self.performance_function == self.NMI:
			results['nmi'] = normalized_mutual_info_score(self.node_labels, self.node_label_predictions)
		elif self.performance_function == self.SILHOUETTE:
			results['silhouette'] = silhouette_score(self.embeddings, self.node_label_predictions,
													 metric='cosine')

		return results
