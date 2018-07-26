# import necessary stuff
import pickle
import numpy as np
import networkx as nx
import argparse
import time
from telegram import Bot
from multiprocessing import Pool, cpu_count


partition_id = 0
ppr_alpha = 0.85

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--ppr_alpha", required=False, type=float,
						help="Specify ppr-alpha restart probability", default=0.85)
	parser.add_argument("--partition_id", required=True, type=int,
						help="Specify partition id you would like to compute node samples for")
	args = parser.parse_args()

	partition_id = args.partition_id
	ppr_alpha = args.ppr_alpha
	print("Compute node sampling for partition {}".format(partition_id))

print("Construct graph")

dataset_path = 'data/coauthor/'
coauthor_crawled_data_file_path = dataset_path + 'coauthor_crawled_data.p'
SEND_NOTIFICATIONS = True

# initialize telegram bot
token = "350553078:AAEu70JDqMFcG_x5eBD3nqccTvc4aFNMKkg"
chat_id = "126551968"
bot = Bot(token)

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
with open(dataset_path + 'coauthor_without_coauthor_edges_networkx.p', 'rb') as pickle_file:
	coauthor_graph = pickle.load(pickle_file)

# define node and edge label constants
AUTHOR = 'author'
PAPER = 'paper'
REFERENCES = 'references'
WRITTEN_BY = 'written_by'

print("{} nodes in graph".format(coauthor_graph.number_of_nodes()))
print("{} edges in graph".format(coauthor_graph.number_of_edges()))

# compute average degree of all nodes in graph
node_degrees = np.array(list(dict(coauthor_graph.degree(list(coauthor_graph.nodes))).values()), dtype=np.int64)
avg_node_degree = np.mean(node_degrees)
print("The avg. node degree is {}".format(np.round(avg_node_degree, decimals=2)))

# define random walk hyper-parameters
sim_G_sampling = {}
samples_per_node = 10000
experiment_name = 'Coauthor without coauthor-edges Partition {} Node Sampling with restart V1'.format(partition_id)

# define meta-path scoring information
meta_path_scheme_A = [AUTHOR, WRITTEN_BY, PAPER, WRITTEN_BY, AUTHOR]
meta_path_scheme_B = [PAPER, WRITTEN_BY, AUTHOR, WRITTEN_BY, PAPER]

meta_path_schemes = {
	AUTHOR: [meta_path_scheme_A],
	PAPER: [meta_path_scheme_B]
}
scoring_function = {}


# sample a meta-path scheme from all meta-path schemes according to given scoring function
def sample_meta_path_scheme(node):
	node_label = coauthor_graph.nodes[node]['label']
	meta_path_scheme_index = np.random.choice(list(range(len(meta_path_schemes[node_label]))))
	meta_path_scheme = meta_path_schemes[node_label][meta_path_scheme_index]

	return meta_path_scheme


# check, whether neighbor (candidate) of node i in walk fulfills requirements given through meta-path scheme
def candidate_valid(node, candidate, meta_path_scheme, step):
	node_label_valid = coauthor_graph.nodes[candidate]['label'] == meta_path_scheme[(step + 1) * 2 - 2]
	edge_label_valid = coauthor_graph[node][candidate]['label'] == meta_path_scheme[(step + 1) * 2 - 3]

	return node_label_valid and edge_label_valid


# compute transition probabilities for all neighborhood nodes of node i according to given meta-path
def compute_candidate_set(meta_path_scheme, step, node):
	candidate_set = list(coauthor_graph[node])
	shrinked_candidate_set = []
	for i, candidate in enumerate(candidate_set):
		if candidate_valid(node, candidate, meta_path_scheme, step):
			shrinked_candidate_set.append(candidate)

	return shrinked_candidate_set


# run single random walk with transition probabilities according to scoring function
def run_single_random_walk(start_node, meta_path_scheme):
	current_node = start_node
	nodes_in_meta_path = int((len(meta_path_scheme) + 1) / 2)

	for i in range(1, nodes_in_meta_path):
		candidate_set = compute_candidate_set(meta_path_scheme, i, current_node)
		if len(candidate_set) == 0:
			return current_node, 0

		current_node = np.random.choice(candidate_set)

	return current_node, 1


# sample 10.000 times a similar node given particular node
def create_samples_for_node(node):
	sampled_nodes = []
	meta_path_scheme = sample_meta_path_scheme(node)
	for i in range(samples_per_node):
		next_node = node
		while np.random.sample(1)[0] < ppr_alpha:
			next_node, has_next_hop = run_single_random_walk(next_node, meta_path_scheme)
			if not has_next_hop:
				break

		sampled_nodes.append(next_node)

	return sampled_nodes


# sample 10.000 similar nodes for each node in node_list in parallel
num_node_partitions = 7724
num_nodes_per_partition = int(coauthor_graph.number_of_nodes() / num_node_partitions)

lower_partition_index = partition_id * num_nodes_per_partition
upper_partition_index = (partition_id + 1) * num_nodes_per_partition
nodes_list = sorted(list(coauthor_graph.nodes))[lower_partition_index:upper_partition_index]
message = "Compute from node {}|{} to {}|{}".format(lower_partition_index, nodes_list[0], upper_partition_index-1,
													nodes_list[-1])
print(message)
try:
	if SEND_NOTIFICATIONS:
		bot.send_message(chat_id=chat_id, text=message)
except:
	print("Failed sending message!")

start_time = time.time()

with Pool(cpu_count()) as pool:
	for i, result in enumerate(pool.imap(create_samples_for_node, nodes_list, chunksize=1)):
		sim_G_sampling[nodes_list[i]] = result
		if (i + 1) % 400 == 0:
			message = "{}: Finished {}/{} nodes".format(experiment_name, i + 1, len(nodes_list))
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

# load id-to-node mapping of verse embeddings
id2node_filepath = dataset_path + 'coauthor_without_coauthor_edges_mapping_ids_to_nodes.p'
with open(id2node_filepath, 'rb') as id_2_node_file:
	id_2_node = pickle.load(id_2_node_file)

# load node-to-id mapping of verse embeddings
node2id_filepath = dataset_path + 'coauthor_without_coauthor_edges_mapping_nodes_to_ids.p'
with open(node2id_filepath, 'rb') as node_2_id_file:
	node_2_id = pickle.load(node_2_id_file)

# save dict with node -> similar-nodes-list as pickle file
export_results_file_path = dataset_path + 'coauthor_without_coauthor_edges_sampling_with_restart_v1_partition_{}.p'.format(partition_id)
with open(export_results_file_path, 'wb') as pickle_file:
	pickle.dump(sim_G_sampling, pickle_file)

message = "Finished storing results of {} ".format(experiment_name)
print(message)
try:
	if SEND_NOTIFICATIONS:
		bot.send_message(chat_id=chat_id, text=message)
except:
	print("Failed sending message!")
