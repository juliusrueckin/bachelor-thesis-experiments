# import necessary stuff and python-wrapper of verse
import pprint
import numpy as np
import pickle
import networkx as nx

from experiment import Experiment

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
coauthor_graph = nx.Graph()

# define node and edge label constants
AUTHOR = 'author'
PAPER = 'paper'
CO_AUTHOR = 'co_author_of'
REFERENCES = 'references'
WRITTEN_BY = 'written_by'

# add authors and papers
for field_of_study in coauthor_data.keys():
    for conference in coauthor_data[field_of_study].keys():
        for year in coauthor_data[field_of_study][conference].keys():
            for i, paper in enumerate(coauthor_data[field_of_study][conference][year]):
                coauthor_graph.add_node('P' + str(paper['Id']), num_citations=paper['CC'], num_references=len(paper['RId']),
                                        conference=conference, field_of_study=field_of_study, label=PAPER)
                for author in coauthor_data[field_of_study][conference][year][i]['authors']:
                    coauthor_graph.add_node('A' + str(author), label=AUTHOR)

print("{} author and paper nodes in graph".format(coauthor_graph.number_of_nodes()))

# add co-author, written_by and reference edge
for field_of_study in coauthor_data.keys():
    for conference in coauthor_data[field_of_study].keys():
        for year in coauthor_data[field_of_study][conference].keys():
            for i, paper in enumerate(coauthor_data[field_of_study][conference][year]):
                for referenced_paper_id in paper['RId']:
                    if 'P' + str(referenced_paper_id) in coauthor_graph:
                        coauthor_graph.add_edge('P' + str(paper['Id']), 'P' + str(referenced_paper_id),
                                                label=REFERENCES)
                for author in coauthor_data[field_of_study][conference][year][i]['authors']:
                    coauthor_graph.add_edge('P' + str(paper['Id']), 'A' + str(author), label=WRITTEN_BY)
                    for co_author in coauthor_data[field_of_study][conference][year][i]['authors']:
                        if author != co_author:
                            coauthor_graph.add_edge('A' + str(author), 'A' + str(co_author), label=CO_AUTHOR)

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

# read *.emb file with precomputed node2vec embeddings
embeddings_file_path = results_path + 'coauthor_deepwalk_embeddings.emb'
embeddings_file = open(embeddings_file_path, "r")
embeddings_file_content = np.fromfile(embeddings_file, dtype=np.float32)
num_of_nodes = int(np.shape(embeddings_file_content)[0] / n_hidden)
deepwalk_embeddings = embeddings_file_content.reshape((num_of_nodes, n_hidden))

# load id-to-node mapping of verse embeddings
id2node_filepath = dataset_path + 'coauthor_mapping_ids_to_nodes.p'
with open(id2node_filepath, 'rb') as id_2_node_file:
    id2node = pickle.load(id_2_node_file)

# load node-to-id mapping of verse embeddings
node2id_filepath = dataset_path + 'coauthor_mapping_nodes_to_ids.p'
with open(node2id_filepath, 'rb') as node_2_id_file:
    node2id = pickle.load(node_2_id_file)

# collect paper train data from verse embeddings
paper_verse_embeddings = []
paper_labels = []
for paper in paper_nodes:
    paper_index = node2id[paper]
    paper_verse_embeddings.append(verse_ppr_embeddings[paper_index])
    paper_labels.append(paper_conference_labels[paper])

# experiment types
CLUSTERING = 'clustering'
CLASSIFICATION = 'classification'
MULTI_LABEL_CLASSIFICATION = 'multi_label_classification'
LINK_PREDICTION = 'link_prediction'

# init classification experiment on verse-ppr embedding
random_seed = 42
num_of_reps = 10
random_seeds = list(range(42, 42+num_of_reps))
train_sizes = [i/20 for i in range(1, num_of_reps+1, 1)]
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