# import necessary stuff
import pickle
import networkx as nx
import pandas as pd
import chardet
from struct import pack

dataset_path = 'data/net_dbis/'
samples_per_node = 10000

# define data set file paths
dataset_path = 'data/net_dbis/'
authors_csv_path = dataset_path + 'id_author.txt'
conferences_csv_path = dataset_path + 'id_conf.txt'
papers_csv_path = dataset_path + 'paper.txt'
paper_author_edges_csv_path = dataset_path + 'paper_author.txt'
paper_conference_edges_csv_path = dataset_path + 'paper_conf.txt'

print("Construct graph")

# detect encodings of files
encodings = {}
file_paths = [authors_csv_path, conferences_csv_path, papers_csv_path, paper_author_edges_csv_path, paper_conference_edges_csv_path]

for file_path in file_paths:
	with open(file_path, 'rb') as f:
		encodings[file_path] = (chardet.detect(f.read()))

# store cvs contents in data frame
authors_df = pd.read_csv(authors_csv_path, sep='\t', header=None, dtype={0: str, 1: str},
						 encoding=encodings[authors_csv_path]["encoding"])
conferences_df = pd.read_csv(conferences_csv_path, sep='\t', header=None, dtype={0: str, 1: str},
							 encoding=encodings[conferences_csv_path]["encoding"])
papers_df = pd.read_csv(papers_csv_path, sep='     ', header=None, dtype={0: str, 1: str},
						encoding=encodings[papers_csv_path]["encoding"])
paper_author_edges_df = pd.read_csv(paper_author_edges_csv_path, sep='\t', header=None, dtype={0: str, 1: str},
									encoding=encodings[paper_author_edges_csv_path]["encoding"])
paper_conference_edges_df = pd.read_csv(paper_conference_edges_csv_path, sep='\t', header=None, dtype={0: str, 1: str},
										encoding=encodings[paper_conference_edges_csv_path]["encoding"])

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

# load id-to-node mapping of verse embeddings
id2node_filepath = dataset_path + 'dbis_mapping_ids_to_nodes.p'
with open(id2node_filepath, 'rb') as id_2_node_file:
	id_2_node = pickle.load(id_2_node_file)

# load node-to-id mapping of verse embeddings
node2id_filepath = dataset_path + 'dbis_mapping_nodes_to_ids.p'
with open(node2id_filepath, 'rb') as node_2_id_file:
	node_2_id = pickle.load(node_2_id_file)

# build nodes x samples_per_node node index matrix for verse c++-implementation
num_node_partitions = 5
num_nodes_per_partition = int(dbis_graph.number_of_nodes() / num_node_partitions)
dbis_sampling_v1_file_path = dataset_path + 'dbis_sampling_v1_partition_{}.p'
node_samples_arr = []
nodes_list = list(dbis_graph.nodes)

for partition_id in range(num_node_partitions):
	print("Start parsing partition {}".format(partition_id))
	dbis_partition_sampling_v1_file_path = dbis_sampling_v1_file_path.format(partition_id)

	with open(dbis_partition_sampling_v1_file_path, 'rb') as pickle_file:
		node_partition_sample_dict = pickle.load(pickle_file)

	node_partition_sample_values = list(node_partition_sample_dict.values())
	flatten_node_partition_sample_values = [node for node_list in node_partition_sample_values for node in node_list]
	node_samples_arr.extend(flatten_node_partition_sample_values)

print("Finished building verse data structure")
print("Start writing to file")

# write node index sample matrix to file
node_index_samples_file_path = dataset_path + 'node_index_samples_dbis_v1.smp'
with open(node_index_samples_file_path, 'wb') as node_index_samples_file:
	node_index_samples_file.write(pack('%di' % len(nodes_list)*samples_per_node, *node_samples_arr))

print("Finished writing to file")
print("Finished paring")
