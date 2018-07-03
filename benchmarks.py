import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class MultiClassClassification:

	start_time = None
	end_time = None

	# algorithm configurations
	method_name = 'Verse-PPR'
	dataset_name = 'Test-Data'
	performance_function = 'Macro-F1'
	train_size = 0.3
	logistic_regression_model = None
	node_label_predictions = []

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

	# initialize classification algorithm with customized configuration parameters and produce random train-test split
	def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='Macro-F1',
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

	def predict(self):
		print('Predict multi-class classification experiment with {} on {} evaluated through {} on {}% train data!'
			.format(self.method_name, self.dataset_name, self.performance_function, self.train_size * 100.00))

		self.start_time = time.time()

		self.node_label_predictions = self.logistic_regression_model.predict(self.embeddings_test)

		self.end_time = time.time()

		total_prediction_time = round(self.end_time - self.start_time, 2)
		print('Predicted multi-class classification experiment in {} sec.!'.format(total_prediction_time))

		return self.node_label_predictions

	# evaluate prediction results through already pre-defined performance function(s)return results as a dict
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
