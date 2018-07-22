# import necessary stuff
import pandas as pd
import networkx as nx
import time
import pickle
import pprint
import chardet
import warnings
import scholarly
import argparse
from telegram import Bot


lower_index = 0
upper_index = 10000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower_index", required=True, type=int, help="Specify lower partition index")
    parser.add_argument("--upper_index", required=True, type=int, help="Specify upper partition index")
    args = parser.parse_args()

    lower_index = args.lower_index
    upper_index = args.upper_index
    print("Crawl google scholar papers from index {} to {}".format(lower_index, upper_index))
    print("Construct Graph")

# ignore warnings
warnings.simplefilter("ignore")

# initialize pretty printer
pp = pprint.PrettyPrinter(indent=4, depth=8)

# initilize telegram bot
token = "350553078:AAEu70JDqMFcG_x5eBD3nqccTvc4aFNMKkg"
chat_id = "126551968"
bot = Bot(token)

# define dataset file paths
dataset_path = 'data/net_dbis/'
authors_csv_path = dataset_path + 'id_author.txt'
conferences_csv_path = dataset_path + 'id_conf.txt'
papers_csv_path = dataset_path + 'paper.txt'
paper_author_edges_csv_path = dataset_path + 'paper_author.txt'
paper_conference_edges_csv_path = dataset_path + 'paper_conf.txt'

#detect encodings of files
encodings = {}
file_paths = [authors_csv_path, conferences_csv_path, papers_csv_path, paper_author_edges_csv_path, paper_conference_edges_csv_path]

for file_path in file_paths:
    with open(file_path, 'rb') as f:
        encodings[file_path] = (chardet.detect(f.read()))

# store cvs contents in dataframe
authors_df = pd.read_csv(authors_csv_path, sep='\t', header=None, dtype={0:str, 1:str}, encoding=encodings[authors_csv_path]["encoding"])
conferences_df = pd.read_csv(conferences_csv_path, sep='\t', header=None, dtype={0:str, 1:str}, encoding=encodings[conferences_csv_path]["encoding"])
papers_df = pd.read_csv(papers_csv_path, sep='     ', header=None, dtype={0:str, 1:str}, encoding=encodings[papers_csv_path]["encoding"])
paper_author_edges_df = pd.read_csv(paper_author_edges_csv_path, sep='\t', header=None, dtype={0:str, 1:str}, encoding=encodings[paper_author_edges_csv_path]["encoding"])
paper_conference_edges_df = pd.read_csv(paper_conference_edges_csv_path, sep='\t', header=None, dtype={0:str, 1:str}, encoding=encodings[paper_conference_edges_csv_path]["encoding"])

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

# create edge tuples from dataframe
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


def crawl_scholar_paper(paper_title):
    cited_by = -1
    result = next(scholarly.search_pubs_query(paper_title))

    if hasattr(result, 'citedby'):
        cited_by = result.citedby

    return cited_by


papers_citations_count = {}
start_time = time.time()

print("Start crawling...")

for i in range(lower_index, upper_index):
    if i % 100 == 0:
        message = "From {} to {}: Already crawled {} papers ".format(lower_index, upper_index, i + 1)
        print(message)
        bot.send_message(chat_id=chat_id, text=message)

    paper_title = papers_df.loc[i, 1].strip()[:-1]
    papers_citations_count[i] = crawl_scholar_paper(paper_title)

end_time = time.time()
crawl_duration = round(end_time - start_time, 2)

message = "Finished crawling google scholar papers from index {} to {}".format(lower_index, upper_index)
print(message)
print("Crawling took {} sec.".format(crawl_duration))

bot.send_message(chat_id=chat_id, text=message)

# save dict with paper-index -> citation count as pickle file
dbis_paper_citation_count_path = dataset_path + 'paper_{}_to_{}_cite_count.p'.format(lower_index, upper_index)
with open(dbis_paper_citation_count_path, 'wb') as pickle_file:
    pickle.dump(papers_citations_count, pickle_file)
