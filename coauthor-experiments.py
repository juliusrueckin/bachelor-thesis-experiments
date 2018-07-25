# import necessary stuff and python-wrapper of verse
import pprint
import numpy as np
import pickle
import networkx as nx
from scipy import stats
from experiment import Experiment

# control runnable experiments
RUN_VERSE_PAPER_CLASSIFICATION = False
RUN_DEEPWALK_PAPER_CLASSIFICATION = False
RUN_NODE2VEC_PAPER_CLASSIFICATION = False
RUN_HETE_VERSE_PAPER_CLASSIFICATION = False
RUN_VERSE_AUTHOR_CLASSIFICATION = True
RUN_DEEPWALK_AUTHOR_CLASSIFICATION = True
RUN_NODE2VEC_AUTHOR_CLASSIFICATION = True
RUN_HETE_VERSE_AUTHOR_CLASSIFICATION = True

# initialize pretty printer
pp = pprint.PrettyPrinter(indent=4, depth=8)

# configure telegram notifier bot
my_telegram_config = {
    "telegram": {
        "token": "350553078:AAEu70JDqMFcG_x5eBD3nqccTvc4aFNMKkg",
        "chat_id": "126551968",
        "verbose": 1
    }
}

dataset_path = 'data/coauthor/'
coauthor_crawled_data_file_path = dataset_path + 'coauthor_crawled_data.p'
EXPORT_AS_EDGE_LIST = False

with open(coauthor_crawled_data_file_path, 'rb') as pickle_file:
    coauthor_data = pickle.load(pickle_file)

# define research fields and years of interest for us
fields_of_studies = ['Machine learning']
years = [2013, 2014, 2015, 2016]

# extract top 5 conferences per field of research
top_5_conf_series_per_field = {}
for field_of_study in fields_of_studies:
    top_5_conf_series_per_field[field_of_study] = coauthor_data[field_of_study]

# define networkx graph
with open(dataset_path + 'coauthor_networkx.p', 'rb') as pickle_file:
    coauthor_graph = pickle.load(pickle_file)

# define node and edge label constants
AUTHOR = 'author'
PAPER = 'paper'
CO_AUTHOR = 'co_author_of'
REFERENCES = 'references'
WRITTEN_BY = 'written_by'

print("{} nodes in graph".format(coauthor_graph.number_of_nodes()))
print("{} edges in graph".format(coauthor_graph.number_of_edges()))

# compute average degree of all nodes in graph
node_degrees = np.array(list(dict(coauthor_graph.degree(list(coauthor_graph.nodes))).values()),dtype=np.int64)
avg_node_degree = np.mean(node_degrees)
print("The avg. node degree is {}".format(np.round(avg_node_degree, decimals=2)))

# collect conference label mapping
conf_count = 0
conference_label_mapping = {}
for field_of_study in coauthor_data.keys():
    for conference in coauthor_data[field_of_study].keys():
        conference_label_mapping[conference] = conf_count
        conf_count += 1

# collect paper nodes 
paper_nodes = [node for node, attr in coauthor_graph.nodes(data=True) if attr['label'] == PAPER]

# collect conference class label for each paper
paper_conference_labels = {}
for paper in paper_nodes:
    paper_conference = coauthor_graph.nodes[paper]['conference']
    paper_conference_labels[paper] = conference_label_mapping[paper_conference]

# read *.emb file with precomputed verse-ppr embeddings
n_hidden = 128
results_path = 'results/coauthor/'
embeddings_file_path = results_path + 'coauthor_verse_ppr_embeddings.emb'
embeddings_file = open(embeddings_file_path, "r")
embeddings_file_content = np.fromfile(embeddings_file, dtype=np.float32)
num_of_nodes = int(np.shape(embeddings_file_content)[0] / n_hidden)
verse_ppr_embeddings = embeddings_file_content.reshape((num_of_nodes, n_hidden))

# read *.emb file with precomputed node2vec embeddings
embeddings_file_path = results_path + 'coauthor_node2vec_embeddings.emb'
embeddings_file = open(embeddings_file_path, "r")
embeddings_file_content = np.fromfile(embeddings_file, dtype=np.float32)
num_of_nodes = int(np.shape(embeddings_file_content)[0] / n_hidden)
node2vec_embeddings = embeddings_file_content.reshape((num_of_nodes, n_hidden))

# read *.emb file with precomputed deepwalk embeddings
embeddings_file_path = results_path + 'coauthor_deepwalk_embeddings.emb'
embeddings_file = open(embeddings_file_path, "r")
embeddings_file_content = np.fromfile(embeddings_file, dtype=np.float32)
num_of_nodes = int(np.shape(embeddings_file_content)[0] / n_hidden)
deepwalk_embeddings = embeddings_file_content.reshape((num_of_nodes, n_hidden))

# read *.emb file with precomputed hete-verse embeddings
embeddings_file_path = results_path + 'coauthor_hete_verse_embeddings.emb'
embeddings_file = open(embeddings_file_path, "r")
embeddings_file_content = np.fromfile(embeddings_file, dtype=np.float32)
num_of_nodes = int(np.shape(embeddings_file_content)[0] / n_hidden)
hete_verse_embeddings = embeddings_file_content.reshape((num_of_nodes, n_hidden))

# load id-to-node mapping of verse embeddings
id2node_filepath = dataset_path + 'coauthor_mapping_ids_to_nodes.p'
with open(id2node_filepath, 'rb') as id_2_node_file:
    id2node = pickle.load(id_2_node_file)

# load node-to-id mapping of verse embeddings
node2id_filepath = dataset_path + 'coauthor_mapping_nodes_to_ids.p'
with open(node2id_filepath, 'rb') as node_2_id_file:
    node2id = pickle.load(node_2_id_file)

# experiment types
CLUSTERING = 'clustering'
CLASSIFICATION = 'classification'
MULTI_LABEL_CLASSIFICATION = 'multi_label_classification'
LINK_PREDICTION = 'link_prediction'

# deine static experiment parameters
random_seed = 42
num_of_reps = 10
random_seeds = list(range(42, 42+num_of_reps))
train_sizes = [i/20 for i in range(1, num_of_reps+1, 1)]

if RUN_VERSE_PAPER_CLASSIFICATION:
    # collect paper train data from verse embeddings
    paper_verse_embeddings = []
    paper_labels = []
    for paper in paper_nodes:
        paper_index = node2id[paper]
        paper_verse_embeddings.append(verse_ppr_embeddings[paper_index])
        paper_labels.append(paper_conference_labels[paper])

    # init classification experiment on verse-ppr embedding
    results_json_path = results_path + 'coauthor_verse_ppr_conference_classification.json'
    results_pickle_path = results_path + 'coauthor_verse_ppr_conference_classification_exp.p'
    coauthor_verse_ppr_classification_experiment = Experiment(method_name='Verse-PPR', dataset_name='co-author', performance_function='both',
                                      node_labels=paper_labels, repetitions=num_of_reps, node_embedings=paper_verse_embeddings,
                                      embedding_dimensionality=n_hidden, experiment_params={'train_size': train_sizes},
                                      results_file_path=results_json_path, experiment_type=CLASSIFICATION,
                                      random_seeds=random_seeds, pickle_path=results_pickle_path,
                                      telegram_config=my_telegram_config)

    # run experiment wrapper: train, predict and evaluate conference classification on verse-ppr embeddings
    coauthor_verse_ppr_classification_experiment_results = coauthor_verse_ppr_classification_experiment.run()

if RUN_NODE2VEC_PAPER_CLASSIFICATION:
    # collect paper train data from node2vec embeddings
    paper_node2vec_embeddings = []
    paper_labels = []
    for paper in paper_nodes:
        paper_index = node2id[paper]
        paper_node2vec_embeddings.append(node2vec_embeddings[paper_index])
        paper_labels.append(paper_conference_labels[paper])

    # init classification experiment on node2vec embedding
    results_json_path = results_path + 'coauthor_node2vec_conference_classification.json'
    results_pickle_path = results_path + 'coauthor_node2vec_conference_classification_exp.p'
    coauthor_node2vec_classification_experiment = Experiment(method_name='node2vec', dataset_name='co-author', performance_function='both',
                                      node_labels=paper_labels, repetitions=num_of_reps, node_embedings=paper_node2vec_embeddings,
                                      embedding_dimensionality=n_hidden, experiment_params={'train_size': train_sizes},
                                      results_file_path=results_json_path, experiment_type=CLASSIFICATION,
                                      random_seeds=random_seeds, pickle_path=results_pickle_path,
                                      telegram_config=my_telegram_config)

    # run experiment wrapper: train, predict and evaluate conference classification on node2vec embeddings
    coauthor_node2vec_classification_experiment_results = coauthor_node2vec_classification_experiment.run()

if RUN_DEEPWALK_PAPER_CLASSIFICATION:
    # collect paper train data from deepwalk embeddings
    paper_deepwalk_embeddings = []
    paper_labels = []
    for paper in paper_nodes:
        paper_index = node2id[paper]
        paper_deepwalk_embeddings.append(deepwalk_embeddings[paper_index])
        paper_labels.append(paper_conference_labels[paper])

    # init classification experiment on deepwalk embedding
    results_json_path = results_path + 'coauthor_deepwalk_conference_classification.json'
    results_pickle_path = results_path + 'coauthor_deepwalk_conference_classification_exp.p'
    coauthor_deepwalk_classification_experiment = Experiment(method_name='deepwalk', dataset_name='co-author', performance_function='both',
                                      node_labels=paper_labels, repetitions=num_of_reps, node_embedings=paper_deepwalk_embeddings,
                                      embedding_dimensionality=n_hidden, experiment_params={'train_size': train_sizes},
                                      results_file_path=results_json_path, experiment_type=CLASSIFICATION,
                                      random_seeds=random_seeds, pickle_path=results_pickle_path,
                                      telegram_config=my_telegram_config)

    # run experiment wrapper: train, predict and evaluate conference classification on deepwalk embeddings
    coauthor_deepwalk_classification_experiment_results = coauthor_deepwalk_classification_experiment.run()

if RUN_HETE_VERSE_PAPER_CLASSIFICATION:
    # collect paper train data from deepwalk embeddings
    paper_hete_verse_embeddings = []
    paper_labels = []
    for paper in paper_nodes:
        paper_index = node2id[paper]
        paper_hete_verse_embeddings.append(hete_verse_embeddings[paper_index])
        paper_labels.append(paper_conference_labels[paper])

    # init classification experiment on deepwalk embedding
    results_json_path = results_path + 'coauthor_hete_verse_conference_classification.json'
    results_pickle_path = results_path + 'coauthor_hete_verse_conference_classification_exp.p'
    coauthor_hete_verse_classification_experiment = Experiment(method_name='hete-VERSE', dataset_name='co-author',
                                                             performance_function='both',
                                                             node_labels=paper_labels, repetitions=num_of_reps,
                                                             node_embedings=paper_hete_verse_embeddings,
                                                             embedding_dimensionality=n_hidden,
                                                             experiment_params={'train_size': train_sizes},
                                                             results_file_path=results_json_path,
                                                             experiment_type=CLASSIFICATION,
                                                             random_seeds=random_seeds, pickle_path=results_pickle_path,
                                                             telegram_config=my_telegram_config)

    # run experiment wrapper: train, predict and evaluate conference classification on deepwalk embeddings
    coauthor_hete_verse_classification_experiment_results = coauthor_hete_verse_classification_experiment.run()

# for all authors, collect all conferences, an autor published in papers
author_nodes = [node for node, attr in coauthor_graph.nodes(data=True) if attr['label'] == AUTHOR]
author_conference_labels = {}
for author in author_nodes:
    author_conference_labels[author] = []
    for neighbor in coauthor_graph[author]:
        if coauthor_graph.nodes[neighbor]['label'] == PAPER:
            author_conference_labels[author].append(coauthor_graph.nodes[neighbor]['conference'])

# for all authors find conference, they published in most papers
for author in author_conference_labels.keys():
    author_conference_labels[author] = stats.mode(author_conference_labels[author]).mode[0]
    author_conference_labels[author] = conference_label_mapping[author_conference_labels[author]]

if RUN_VERSE_AUTHOR_CLASSIFICATION:
    # collect author train data from verse-ppr embeddings
    author_verse_embeddings = []
    author_labels = []
    for author in author_nodes:
        author_index = node2id[author]
        author_verse_embeddings.append(verse_ppr_embeddings[author_index])
        author_labels.append(author_conference_labels[author])

    # init classification experiment on verse-ppr embedding
    results_json_path = results_path + 'coauthor_verse_ppr_author_conference_classification.json'
    results_pickle_path = results_path + 'coauthor_verse_ppr_author_conference_classification_exp.p'
    coauthor_verse_ppr_classification_experiment = Experiment(method_name='Verse-PPR', dataset_name='co-author',
                                                              performance_function='both',
                                                              node_labels=author_labels, repetitions=num_of_reps,
                                                              node_embedings=author_verse_embeddings,
                                                              embedding_dimensionality=n_hidden,
                                                              experiment_params={'train_size': train_sizes},
                                                              results_file_path=results_json_path,
                                                              experiment_type=CLASSIFICATION,
                                                              random_seeds=random_seeds,
                                                              pickle_path=results_pickle_path,
                                                              telegram_config=my_telegram_config)

    # run experiment wrapper: train, predict and evaluate conference classification on verse-ppr embeddings
    coauthor_verse_ppr_classification_experiment_results = coauthor_verse_ppr_classification_experiment.run()

if RUN_NODE2VEC_AUTHOR_CLASSIFICATION:
    # collect author train data from node2vec embeddings
    author_node2vec_embeddings = []
    author_labels = []
    for author in author_nodes:
        author_index = node2id[author]
        author_node2vec_embeddings.append(node2vec_embeddings[author_index])
        author_labels.append(author_conference_labels[author])

    # init classification experiment on node2vec embedding
    results_json_path = results_path + 'coauthor_node2vec_author_conference_classification.json'
    results_pickle_path = results_path + 'coauthor_node2vec_author_conference_classification_exp.p'
    coauthor_node2vec_classification_experiment = Experiment(method_name='node2vec', dataset_name='co-author',
                                                              performance_function='both',
                                                              node_labels=author_labels, repetitions=num_of_reps,
                                                              node_embedings=author_node2vec_embeddings,
                                                              embedding_dimensionality=n_hidden,
                                                              experiment_params={'train_size': train_sizes},
                                                              results_file_path=results_json_path,
                                                              experiment_type=CLASSIFICATION,
                                                              random_seeds=random_seeds,
                                                              pickle_path=results_pickle_path,
                                                              telegram_config=my_telegram_config)

    # run experiment wrapper: train, predict and evaluate conference classification on node2vec embeddings
    coauthor_node2vec_classification_experiment_results = coauthor_node2vec_classification_experiment.run()

if RUN_DEEPWALK_AUTHOR_CLASSIFICATION:
    # collect author train data from deepwalk embeddings
    author_deepwalk_embeddings = []
    author_labels = []
    for author in author_nodes:
        author_index = node2id[author]
        author_deepwalk_embeddings.append(deepwalk_embeddings[author_index])
        author_labels.append(author_conference_labels[author])

    # init classification experiment on deepwalk embedding
    results_json_path = results_path + 'coauthor_deepwalk_author_conference_classification.json'
    results_pickle_path = results_path + 'coauthor_deepwalk_author_conference_classification_exp.p'
    coauthor_deepwalk_classification_experiment = Experiment(method_name='deepwalk', dataset_name='co-author',
                                                              performance_function='both',
                                                              node_labels=author_labels, repetitions=num_of_reps,
                                                              node_embedings=author_deepwalk_embeddings,
                                                              embedding_dimensionality=n_hidden,
                                                              experiment_params={'train_size': train_sizes},
                                                              results_file_path=results_json_path,
                                                              experiment_type=CLASSIFICATION,
                                                              random_seeds=random_seeds,
                                                              pickle_path=results_pickle_path,
                                                              telegram_config=my_telegram_config)

    # run experiment wrapper: train, predict and evaluate conference classification on deepwalk embeddings
    coauthor_deepwalk_classification_experiment_results = coauthor_deepwalk_classification_experiment.run()

if RUN_HETE_VERSE_AUTHOR_CLASSIFICATION:
    # collect author train data from hete-verse embeddings
    author_hete_verse_embeddings = []
    author_labels = []
    for author in author_nodes:
        author_index = node2id[author]
        author_hete_verse_embeddings.append(hete_verse_embeddings[author_index])
        author_labels.append(author_conference_labels[author])

    # init classification experiment on hete-verse embedding
    results_json_path = results_path + 'coauthor_hete_verse_author_conference_classification.json'
    results_pickle_path = results_path + 'coauthor_hete_verse_author_conference_classification_exp.p'
    coauthor_hete_verse_classification_experiment = Experiment(method_name='hete-verse', dataset_name='co-author',
                                                              performance_function='both',
                                                              node_labels=author_labels, repetitions=num_of_reps,
                                                              node_embedings=author_hete_verse_embeddings,
                                                              embedding_dimensionality=n_hidden,
                                                              experiment_params={'train_size': train_sizes},
                                                              results_file_path=results_json_path,
                                                              experiment_type=CLASSIFICATION,
                                                              random_seeds=random_seeds,
                                                              pickle_path=results_pickle_path,
                                                              telegram_config=my_telegram_config)

    # run experiment wrapper: train, predict and evaluate conference classification on hete-verse embeddings
    coauthor_hete_verse_classification_experiment_results = coauthor_hete_verse_classification_experiment.run()
