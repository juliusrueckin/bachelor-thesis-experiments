import numpy as np
import pandas as pd
import networkx as nx
import pickle
from struct import pack

EXPORT_AS_EDGE_LIST = False

# define dataset file paths
dataset_path = 'data/BlogCatalog-dataset/data/'
friend_edges_csv_path = dataset_path + 'edges.csv'
group_edges_csv_path = dataset_path + 'group-edges.csv'
groups_csv_path = dataset_path + 'groups.csv'
bloggers_csv_path = dataset_path + 'nodes.csv'


# In[6]:


# store cvs contents in dataframe
friend_edges_df = pd.read_csv(friend_edges_csv_path, sep=',', header=None, dtype={0: str, 1:str})
group_edges_df = pd.read_csv(group_edges_csv_path, sep=',', header=None, dtype={0: str, 1:str})
groups_df = pd.read_csv(groups_csv_path, sep=',', header=None, dtype={0: str})
bloggers_df = pd.read_csv(bloggers_csv_path, sep=',', header=None, dtype={0: str})


# In[7]:


# give bloggers and groups unique node-ids
bloggers_df[0] = 'b' + bloggers_df[0]
friend_edges_df = 'b' + friend_edges_df
groups_df[0] = 'g' + groups_df[0]
group_edges_df[0] = 'b' + group_edges_df[0]
group_edges_df[1] = 'g' + group_edges_df[1]


# In[8]:


# define networkx graph
blog_catalog_graph = nx.Graph()


# In[9]:


# define node and edge label constants
IS_MEMBER_OF = 'is_member_of'
IS_FRIEND_WITH = 'is_friend_with'
BLOGGER = 'blogger'
GROUP = 'group'


# In[10]:


# add blogger and group nodes to graph
blog_catalog_graph.add_nodes_from(bloggers_df[0].tolist(), label=BLOGGER)
print("{} nodes in graph".format(blog_catalog_graph.number_of_nodes()))
blog_catalog_graph.add_nodes_from(groups_df[0].tolist(), label=GROUP)
print("{} nodes in graph".format(blog_catalog_graph.number_of_nodes()))


# In[11]:


# create edge tuples from dataframe
group_edges = list(zip(group_edges_df[0].tolist(), group_edges_df[1].tolist()))
friend_edges = list(zip(friend_edges_df[0].tolist(), friend_edges_df[1].tolist()))


# In[12]:


# add (blogger)-[is_member_of]-(group) edges to graph
blog_catalog_graph.add_edges_from(group_edges, label=IS_MEMBER_OF)
print("{} edges in graph".format(blog_catalog_graph.number_of_edges()))
print("{} nodes in graph".format(blog_catalog_graph.number_of_nodes()))


# In[13]:


# add (blogger)-[is_friend_with]-(blogger) edges to graph
blog_catalog_graph.add_edges_from(friend_edges, label=IS_FRIEND_WITH)
print("{} edges in graph".format(blog_catalog_graph.number_of_edges()))
print("{} nodes in graph".format(blog_catalog_graph.number_of_nodes()))


# In[14]:


# export graph as edge list to given path
if EXPORT_AS_EDGE_LIST:
    edge_list_export_path = dataset_path + 'blogcatalog_edgelist.csv'
    nx.write_edgelist(blog_catalog_graph, edge_list_export_path, data=False)


# In[15]:

# compute average degree of all nodes in graph
node_degrees = np.array(list(dict(blog_catalog_graph.degree(list(blog_catalog_graph.nodes))).values()),dtype=np.int64)
avg_node_degree = np.mean(node_degrees)
print("The avg. node degree is {}".format(np.round(avg_node_degree, decimals=2)))

# read dict with node-id -> similar-nodes-list from pickle file
blogcatalog_sampling_v1_file_path = 'results/blogcatalog/blogcatalog_sampling_v1.p'
sim_G_sampling_reload={}
with open(blogcatalog_sampling_v1_file_path, 'rb') as pickle_file:
    sim_G_sampling_reload = pickle.load(pickle_file)

# load id-to-node mapping of verse embeddings
id2node_filepath = 'data/BlogCatalog-dataset/data/blogcatalog_mapping_ids_to_nodes.p'
id_2_node = {}
with open(id2node_filepath, 'rb') as id_2_node_file:
    id_2_node = pickle.load(id_2_node_file)

# load node-to-id mapping of verse embeddings
node2id_filepath = 'data/BlogCatalog-dataset/data/blogcatalog_mapping_nodes_to_ids.p'
node_2_id = {}
with open(node2id_filepath, 'rb') as node_2_id_file:
    node_2_id = pickle.load(node_2_id_file)

# build nodes x samples_per_node node index matrix for verse c++-implementation
nodes_list = list(blog_catalog_graph.nodes)
node_samples_arr = []
for i in range((len(nodes_list))):
    node = id_2_node[i]
    sampled_nodes = sim_G_sampling_reload[node]
    sampled_node_indices = []
    for n in sim_G_sampling_reload[node]:
        sampled_node_indices.append(node_2_id[n])
    node_samples_arr.extend(sampled_node_indices)

# write node index sample matrix to file
samples_per_node = 10000
node_index_samples_file_path = dataset_path + 'node_index_samples_blogcatalog_v1.smp'
with open(node_index_samples_file_path, 'wb') as node_index_samples_file:
    node_index_samples_file.write(pack('%di' % len(nodes_list)*samples_per_node, *node_samples_arr))
