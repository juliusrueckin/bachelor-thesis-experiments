import numpy as np
import networkx as nx
import pickle
from telegram import Bot
from struct import pack

print("Construct graph")

dataset_path = 'data/coauthor/'
coauthor_crawled_data_file_path = dataset_path + 'coauthor_json.p'
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

# load id-to-node mapping of verse embeddings
id2node_filepath = dataset_path + 'coauthor_mapping_ids_to_nodes.p'
with open(id2node_filepath, 'rb') as id_2_node_file:
	id_2_node = pickle.load(id_2_node_file)

# load node-to-id mapping of verse embeddings
node2id_filepath = dataset_path + 'coauthor_mapping_nodes_to_ids.p'
with open(node2id_filepath, 'rb') as node_2_id_file:
	node_2_id = pickle.load(node_2_id_file)

# read dict with node-id -> similar-nodes-list from pickle file
num_node_partitions = 4
num_nodes_per_partition = int(coauthor_graph.number_of_nodes() / num_node_partitions)
sim_G_sampling_reload = {}
for partition_id in range(num_node_partitions):
	message = "Coauthor parsing: Load partition {}".format(partition_id)
	print(message)
	try:
		if SEND_NOTIFICATIONS:
			bot.send_message(chat_id=chat_id, text=message)
	except:
		print("Failed sending message!")

	coauthor_partition_sampling_v1_file_path = dataset_path + 'coauthor_sampling_v1_partition_{}.p'.format(partition_id)
	with open(coauthor_partition_sampling_v1_file_path, 'rb') as pickle_file:
		sim_G_partition_sampling_reload = pickle.load(pickle_file)
		message = "Length of partition dict: {}\n".format(len(sim_G_partition_sampling_reload.keys()))
		message += "First node of partition: {}\n".format(list(sim_G_partition_sampling_reload.keys())[0])
		message += "Last node of partition: {}\n".format(list(sim_G_partition_sampling_reload.keys())[-1])
		sim_G_sampling_reload = {**sim_G_sampling_reload, **sim_G_partition_sampling_reload}
		message += "Length of dict: {}\n".format(len(sim_G_sampling_reload.keys()))
		message += "Next first node of dict: {}\n".format(list(sim_G_sampling_reload.keys())[partition_id*num_nodes_per_partition])
		message += "Last node of dict: {}\n".format(list(sim_G_sampling_reload.keys())[-1])
		print(message)
		try:
			if SEND_NOTIFICATIONS:
				bot.send_message(chat_id=chat_id, text=message)
		except:
			print("Failed sending message!")

message = "Build node index matrix for verse"
print(message)
try:
	if SEND_NOTIFICATIONS:
		bot.send_message(chat_id=chat_id, text=message)
except:
	print("Failed sending message!")

# build nodes x samples_per_node node index matrix for verse c++-implementation
nodes_list = sorted(list(coauthor_graph.nodes))
node_samples_arr = []
for i in range((len(nodes_list))):
	node = id_2_node[i]
	sampled_nodes = sim_G_sampling_reload[node]
	sampled_node_indices = []
	for n in sim_G_sampling_reload[node]:
		sampled_node_indices.append(node_2_id[n])
	node_samples_arr.extend(sampled_node_indices)

message = "Coauthor parsing: Write node index matrix to file"
print(message)
try:
	if SEND_NOTIFICATIONS:
		bot.send_message(chat_id=chat_id, text=message)
except:
	print("Failed sending message!")

# write node index sample matrix to file
samples_per_node = 10000
node_index_samples_file_path = dataset_path + 'node_index_samples_coauthor_v1.smp'
with open(node_index_samples_file_path, 'wb') as node_index_samples_file:
	node_index_samples_file.write(pack('%di' % len(nodes_list)*samples_per_node, *node_samples_arr))

message = "Finished coauthor parsing"
print(message)
try:
	if SEND_NOTIFICATIONS:
		bot.send_message(chat_id=chat_id, text=message)
except:
	print("Failed sending message!")
