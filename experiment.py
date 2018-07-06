# import required packages
import numpy as np
import json
from itertools import product
from multi_class_classification import MultiClassClassification
from multi_label_classification import MultiLabelClassification
from clustering import Clustering
from link_prediction import LinkPrediction


class Experiment:

    # experiment types
    CLUSTERING = 'clustering'
    CLASSIFICATION = 'classification'
    MULTI_LABEL_CLASSIFICATION = 'multi_label_classification'
    LINK_PREDICTION = 'link_prediction'

    def __init__(self, method_name='Verse-PPR', dataset_name='Test-Data', performance_function='both', node_labels=[],
                 embeddings_file_path='', node_embedings=None, embedding_dimensionality=128, repetitions=10,
                 experiment_params={}, experiment_type='clustering', results_file_path=None, random_seeds=None):
        """
        Initialize experiment with given configuration parameters
        :param method_name:
        :param dataset_name:
        :param performance_function:
        :param node_labels:
        :param embeddings_file_path:
        :param node_embedings:
        :param embedding_dimensionality:
        :param repetitions:
        :param experiment_params:
        :param experiment_type:
        :param results_file_path:
        :param random_seeds:
        """
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.performance_function = performance_function
        self.embeddings_file_path = embeddings_file_path
        self.node_embeddings = node_embedings
        self.embedding_dimensionality = embedding_dimensionality
        self.node_labels = node_labels
        self.repetitions = repetitions
        self.experiment_params = experiment_params
        self.experiment_type = experiment_type
        self.results_file_path = results_file_path
        self.random_seed = random_seeds
        self.experiment = None
        self.random_seeds = random_seeds

        assert len(self.random_seed) == self.repetitions, 'random seed array length and number of ' \
                                                          'repetitions are not equal'

        self.generate_cross_product_params()

        if self.node_embeddings is None:
            self.read_node_embeddings_from_binary_file()

        self.experiment_results = {
            'method': self.method_name,
            'dataset': self.dataset_name,
            'embedding_file': self.embeddings_file_path,
            'repetitions': self.repetitions,
            'parameterizations': []
        }

    def generate_results_file(self):
        with open(self.results_file_path, 'w') as results_file:
            results_file.write(json.dumps(self.experiment_results, ensure_ascii=False, indent=4))

    def generate_cross_product_params(self):
        cross_product_experiment_params = []
        for values in product(*self.experiment_params.values()):
            cross_product_experiment_params.append(dict(zip(self.experiment_params.keys(), values)))

        self.experiment_params = cross_product_experiment_params

    def read_node_embeddings_from_binary_file(self):
        """
        Read given binary file and convert it to numpy (num_of_nodes, embedding_dimensions) shaped embeddings matrix
        :return:
        """
        embeddings_file = open(self.embeddings_file_path, "r")
        embeddings_file_content = np.fromfile(embeddings_file, dtype=np.float32)
        num_of_nodes = int(np.shape(embeddings_file_content)[0] / self.embedding_dimensionality)
        self.node_embeddings = embeddings_file_content.reshape((num_of_nodes, self.embedding_dimensionality))

    def init_run(self, run_params, random_seed):
        if self.experiment_type == self.CLASSIFICATION:
            self.experiment = \
                MultiClassClassification(method_name=self.method_name, dataset_name=self.dataset_name,
                                         performance_function=self.performance_function,
                                         embeddings=self.node_embeddings,
                                         node_labels=self.node_labels, random_seed=random_seed)
        elif self.experiment_type == self.CLUSTERING:
            self.experiment = \
                Clustering(method_name=self.method_name, dataset_name=self.dataset_name,
                           embeddings=self.node_embeddings, random_seed=random_seed, **run_params,
                           performance_function=self.performance_function, node_labels=self.node_labels)
        elif self.experiment_type == self.MULTI_LABEL_CLASSIFICATION:
            self.experiment = \
                MultiLabelClassification(method_name=self.method_name, dataset_name=self.dataset_name,
                                         node_labels=self.node_labels, random_seed=random_seed,
                                         performance_function=self.performance_function,
                                         embeddings=self.node_embeddings, **run_params)
        elif self.experiment_type == self.LINK_PREDICTION:
            self.experiment = \
                LinkPrediction(method_name=self.method_name, dataset_name=self.dataset_name,
                               node_embeddings=self.node_embeddings, random_seed=random_seed,
                               performance_function=self.performance_function, **run_params)

    def run(self):
        print('Start {} experiment on {} data set with {} embeddings\nRepeated {} times and evaluated through {}'
              'performance function(s)'.format(self.experiment_type, self.dataset_name, self.method_name,
                                               self.repetitions, self.performance_function))

        for index, run_params in enumerate(self.experiment_params):
            self.experiment_results['parameterizations'].append({
                'params': run_params,
                'runs': []
            })

            for rep in range(self.repetitions):
                self.init_run(run_params, self.random_seeds[rep])

                self.experiment.train()
                predictions = self.experiment.predict()
                evaluation = self.experiment.evaluate()

                run_results = {
                    'run': rep + 1,
                    'random_seed': self.random_seeds[rep],
                    'predictions': predictions.tolist(),
                    'evaluation': evaluation
                }

                self.experiment_results['parameterizations'][index]['runs'].append(run_results)

        print('Finished {} experiment on {} data set with {} embeddings'
              .format(self.experiment_type, self.dataset_name, self.method_name))

        if self.results_file_path is not None:
            self.generate_results_file()
            print('Saved results in file {}'.format(self.results_file_path))

        return self.experiment_results
