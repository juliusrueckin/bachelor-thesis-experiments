# import necessary stuff
import numpy as np
import pandas as pd
import networkx as nx
import time
import pickle
import pprint
import chardet
import argparse
from telegram import Bot
from multiprocessing import Pool, cpu_count
from heapq import nlargest

partition_id = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_id", required=True, type=int,
                        help="Specify partition id you would like to compute node samples for")
    args = parser.parse_args()

    partition_id = args.partition_id
    print("Compute node sampling for partition {}".format(partition_id))

print("Construct graph")

EXPORT_AS_EDGE_LIST = True
EXTRACT_SUB_GRAPH = False
SEND_NOTIFICATIONS = False

# initialize pretty printer
pp = pprint.PrettyPrinter(indent=4, depth=8)

# initialize telegram bot
token = "350553078:AAEu70JDqMFcG_x5eBD3nqccTvc4aFNMKkg"
chat_id = "126551968"
bot = Bot(token)

# define data set file paths
dataset_path = 'data/net_dbis/'
authors_csv_path = dataset_path + 'id_author.txt'
conferences_csv_path = dataset_path + 'id_conf.txt'
papers_csv_path = dataset_path + 'paper.txt'
paper_author_edges_csv_path = dataset_path + 'paper_author.txt'
paper_conference_edges_csv_path = dataset_path + 'paper_conf.txt'

# detect encodings of files
encodings = {}
file_paths = [authors_csv_path, conferences_csv_path, papers_csv_path, paper_author_edges_csv_path, paper_conference_edges_csv_path]

for file_path in file_paths:
    with open(file_path, 'rb') as f:
        encodings[file_path] = (chardet.detect(f.read()))

# store cvs contents in data frame
authors_df = pd.read_csv(authors_csv_path, sep='\t', header=None, dtype={0: str, 1: str}, engine='python',
                         encoding=encodings[authors_csv_path]["encoding"])
conferences_df = pd.read_csv(conferences_csv_path, sep='\t', header=None, dtype={0: str, 1: str}, engine='python',
                             encoding=encodings[conferences_csv_path]["encoding"])
papers_df = pd.read_csv(papers_csv_path, sep='     ', header=None, dtype={0: str, 1: str}, engine='python',
                        encoding=encodings[papers_csv_path]["encoding"])
paper_author_edges_df = pd.read_csv(paper_author_edges_csv_path, sep='\t', header=None, dtype={0: str, 1: str},
                                    encoding=encodings[paper_author_edges_csv_path]["encoding"])
paper_conference_edges_df = pd.read_csv(paper_conference_edges_csv_path, sep='\t', header=None, dtype={0: str, 1: str},
                                        encoding=encodings[paper_conference_edges_csv_path]["encoding"], engine='python')

# give authors, papers and conferences unique node-ids
authors_df[0] = 'a' + authors_df[0]
conferences_df[0] = 'c' + conferences_df[0]
papers_df[0] = 'p' + papers_df[0]
paper_author_edges_df[0] = 'p' + paper_author_edges_df[0]
paper_author_edges_df[1] = 'a' + paper_author_edges_df[1]
paper_conference_edges_df[0] = 'p' + paper_conference_edges_df[0]
paper_conference_edges_df[1] = 'c' + paper_conference_edges_df[1]

# define networkx graph
dbis_graph = nx.Graph()

# define node and edge label constants
AUTHOR = 'author'
PAPER = 'paper'
CONFERENCE = 'conference'
PUBLISHED_AT = 'published_at'
WRITTEN_BY = 'written_by' 

# add author, paper and conference nodes to graph
dbis_graph.add_nodes_from(authors_df[0].tolist(), label=AUTHOR)
print("{} nodes in graph".format(dbis_graph.number_of_nodes()))
dbis_graph.add_nodes_from(papers_df[0].tolist(), label=PAPER)
print("{} nodes in graph".format(dbis_graph.number_of_nodes()))
dbis_graph.add_nodes_from(conferences_df[0].tolist(), label=CONFERENCE)
print("{} nodes in graph".format(dbis_graph.number_of_nodes()))

# create edge tuples from data frame
paper_author_edges = list(zip(paper_author_edges_df[0].tolist(), paper_author_edges_df[1].tolist()))
paper_conference_edges = list(zip(paper_conference_edges_df[0].tolist(), paper_conference_edges_df[1].tolist()))

# add (paper)-[published_at]-(conference) edges to graph
dbis_graph.add_edges_from(paper_conference_edges, label=PUBLISHED_AT)
print("{} edges in graph".format(dbis_graph.number_of_edges()))
print("{} nodes in graph".format(dbis_graph.number_of_nodes()))

# add (paper)-[written_by]-(author) edges to graph
dbis_graph.add_edges_from(paper_author_edges, label=WRITTEN_BY)
print("{} edges in graph".format(dbis_graph.number_of_edges()))
print("{} nodes in graph".format(dbis_graph.number_of_nodes()))

# extract top-5000 authors with regard to number of publications
# add each author with less than 8 papers to the delete candidates
if EXTRACT_SUB_GRAPH:
    num_of_top_k_authors = 5000
    author_degrees = []
    for node in list(dbis_graph.nodes):
        if dbis_graph.nodes[node]['label'] == AUTHOR:
            author_degrees.append(dbis_graph.degree(node))

    top_k_author_degree_threshold = min(nlargest(num_of_top_k_authors, author_degrees))
    delete_candidates = []

    for node in list(dbis_graph.nodes):
        if dbis_graph.nodes[node]['label'] == AUTHOR:
            if dbis_graph.degree(node) <= top_k_author_degree_threshold:
                delete_candidates.append(node)

    print("{} authors with less than {} papers are delete candidates".format(len(delete_candidates),
                                                                             top_k_author_degree_threshold+1))

# remove author delete candidates from graph
if EXTRACT_SUB_GRAPH:
    dbis_graph.remove_nodes_from(delete_candidates)
    print("{} edges in graph".format(dbis_graph.number_of_edges()))
    print("{} nodes in graph".format(dbis_graph.number_of_nodes()))

# export graph as edge list to given path
if EXPORT_AS_EDGE_LIST:
    edge_list_export_path = dataset_path + 'dbis_edgelist.csv'
    nx.write_edgelist(dbis_graph, edge_list_export_path, data=False)

# compute average degree of all nodes in graph
node_degrees = np.array(list(dict(dbis_graph.degree(list(dbis_graph.nodes))).values()),dtype=np.int64)
avg_node_degree = np.mean(node_degrees)
print("The avg. node degree is {}".format(np.round(avg_node_degree, decimals=2)))

# define random walk hyper-parameters
sim_G_sampling = {}
samples_per_node = 10000
finished_nodes = 0
experiment_name = 'DBIS Partition {} Node Sampling V1'.format(partition_id)

# define meta-path scoring information
meta_path_scheme_A = [AUTHOR, WRITTEN_BY, PAPER, WRITTEN_BY, AUTHOR]
meta_path_scheme_B = [AUTHOR, WRITTEN_BY, PAPER, PUBLISHED_AT, CONFERENCE, PUBLISHED_AT, PAPER, WRITTEN_BY, AUTHOR]
meta_path_scheme_C = [PAPER, WRITTEN_BY, AUTHOR, WRITTEN_BY, PAPER]
meta_path_scheme_D = [PAPER, PUBLISHED_AT, CONFERENCE, PUBLISHED_AT, PAPER]
meta_path_scheme_E = [PAPER, WRITTEN_BY, AUTHOR, WRITTEN_BY, PAPER, WRITTEN_BY, AUTHOR, WRITTEN_BY, PAPER]
meta_path_scheme_F = [PAPER, WRITTEN_BY, AUTHOR, WRITTEN_BY, PAPER, WRITTEN_BY, AUTHOR, WRITTEN_BY, PAPER, WRITTEN_BY, AUTHOR, WRITTEN_BY, PAPER]
meta_path_scheme_G = [CONFERENCE, PUBLISHED_AT, PAPER, WRITTEN_BY, AUTHOR, WRITTEN_BY, PAPER, PUBLISHED_AT, CONFERENCE]

meta_path_schemes = {
    AUTHOR: [meta_path_scheme_A, meta_path_scheme_B],
    PAPER: [meta_path_scheme_C, meta_path_scheme_D, meta_path_scheme_E, meta_path_scheme_F],
    CONFERENCE: [meta_path_scheme_G]}
scoring_function = {}


# sample a meta-path scheme from all meta-path schemes according to given scoring function
def sample_meta_path_scheme(node):
    node_label = dbis_graph.nodes[node]['label']
    meta_path_scheme_index = np.random.choice(list(range(len(meta_path_schemes[node_label]))))
    meta_path_scheme = meta_path_schemes[node_label][meta_path_scheme_index]
    
    return meta_path_scheme


# check, whether neighbor (candidate) of node i in walk fulfills requirements given through meta-path scheme
def candidate_valid(node, candidate, meta_path_scheme,step):
    node_label_valid = dbis_graph.nodes[candidate]['label'] == meta_path_scheme[(step+1)*2-2]
    edge_label_valid = dbis_graph[node][candidate]['label'] == meta_path_scheme[(step+1)*2-3]
    
    return node_label_valid and edge_label_valid


# compute transition probabilities for all neighborhood nodes of node i according to given meta-path
def compute_candidate_set(meta_path_scheme, step, node):
    candidate_set = list(dbis_graph[node])
    shrinked_candidate_set = []
    for i, candidate in enumerate(candidate_set):
        if candidate_valid(node, candidate, meta_path_scheme, step):
            shrinked_candidate_set.append(candidate)
    
    return shrinked_candidate_set


# run single random walk with transition probabilities according to scoring function
def run_single_random_walk(start_node):
    current_node = start_node
    meta_path_scheme = sample_meta_path_scheme(start_node)
    nodes_in_meta_path = int((len(meta_path_scheme) + 1) / 2)

    for i in range(1, nodes_in_meta_path):
        candidate_set = compute_candidate_set(meta_path_scheme, i, current_node)
        if len(candidate_set) == 0:
            return current_node

        current_node = np.random.choice(candidate_set)

    return current_node


# sample 10.000 times a similar node given particular node
def create_samples_for_node(node):
    sampled_nodes = []
    
    for i in range(samples_per_node):
        sampled_nodes.append(run_single_random_walk(node))
        
    return sampled_nodes


# sample 10.000 similar nodes for each node in node_list in parallel
num_node_partitions = 5
num_nodes_per_partition = int(dbis_graph.number_of_nodes() / num_node_partitions)

lower_partition_index = partition_id * num_nodes_per_partition
upper_partition_index = (partition_id + 1) * num_nodes_per_partition
nodes_list = list(dbis_graph.nodes)[lower_partition_index:upper_partition_index]
start_time = time.time()

with Pool(cpu_count()) as pool:
    for i, result in enumerate(pool.imap(create_samples_for_node, nodes_list, chunksize=1)):
        sim_G_sampling[nodes_list[i]] = result
        if (i+1) % 400 == 0:
            message = "{}: Finished {}/{} nodes".format(experiment_name, i+1, len(nodes_list))
            print(message)
            try:
                if SEND_NOTIFICATIONS:
                    bot.send_message(chat_id=chat_id, text=message)
            except:
                print("Failed sending message!")
        
end_time = time.time()
computation_time = end_time - start_time
print("Whole sampling process took {} sec.".format(np.around(computation_time, decimals=2)))
try:
    if SEND_NOTIFICATIONS:
        bot.send_message(chat_id=chat_id, text="Finished {}: sampling {} nodes for each of {} nodes"
                         .format(experiment_name, samples_per_node, len(nodes_list)))
except:
    print("Failed sending message!")

# save dict with node-id -> similar-nodes-list as pickle file
dbis_sampling_v1_file_path = dataset_path + 'dbis_sampling_v1_partition_{}.p'.format(partition_id)
with open(dbis_sampling_v1_file_path, 'wb') as pickle_file:
    pickle.dump(sim_G_sampling, pickle_file)
