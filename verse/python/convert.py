#   encoding: utf8
#   convert.py
"""Converter for three common graph formats (MATLAB sparse matrix, adjacency
list, edge list) can be found in the root directory of the project.
"""

from collections import defaultdict
from scipy.io import loadmat
from struct import pack

import click
import logging
import numpy as np
import pickle


MAGIC = 'XGFS'.encode('utf8')


def mat2xgfs(filename, undirected, varname):
    mat = loadmat(filename)[varname].tocsr()
    if undirected:
        mat = mat + mat.T  # we dont care about weights in this implementation
    return mat.indptr[:-1], mat.indices


def xgfs2file(outf, indptr, indices):
    nv = indptr.size
    ne = indices.size
    logging.info('num vertices=%d; num edges=%d;', nv, ne)
    outf.write(MAGIC)
    outf.write(pack('q', nv))
    outf.write(pack('q', ne))
    outf.write(pack('%di' % nv, *indptr))
    outf.write(pack('%di' % ne, *indices))


def is_numbers_only(nodes):
    for node in nodes:
        try:
            int(node)
        except ValueError:
            return False
    return True


def list2mat(input, undirected, sep, filepath_pickle):
    nodes = read_nodes_from_file(input, sep)
    isnumbers, node2id, number_of_nodes = map_nodes_to_ids(nodes, filepath_pickle)
    map_ids_to_nodes(nodes, filepath_pickle)
    graph = build_graph(input, sep, node2id, undirected, isnumbers)
    indptr = np.zeros(number_of_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    for i in range(number_of_nodes):
        indptr[i + 1] = indptr[i] + len(graph[i])
    number_of_edges = indptr[-1]
    indices = np.zeros(number_of_edges, dtype=np.int32)
    cur = 0
    for node in range(number_of_nodes):
        for adjv in sorted(graph[node]):
            indices[cur] = adjv
            cur += 1
    return indptr[:-1], indices


def map_nodes_to_ids(nodes: set, filepath_pickle: str = None):
    number_of_nodes = len(nodes)
    isnumbers = is_numbers_only(nodes)
    logging.info('Node IDs are numbers: %s', isnumbers)
    if isnumbers:
        node2id = dict(zip(sorted(map(int, nodes)), range(number_of_nodes)))
    else:
        node2id = dict(zip(sorted(nodes), range(number_of_nodes)))
    if filepath_pickle is not None:
        filepath_pickle = filepath_pickle + '_nodes_to_ids.p'
        with open(filepath_pickle, 'wb') as file:
            pickle.dump(node2id, file)
    return isnumbers, node2id, number_of_nodes


def map_ids_to_nodes(nodes: set, filepath_pickle: str = None):
    number_of_nodes = len(nodes)
    isnumbers = is_numbers_only(nodes)
    logging.info('Node IDs are numbers: %s', isnumbers)
    if isnumbers:
        id2node = dict(zip(range(number_of_nodes), sorted(map(int, nodes))))
    else:
        id2node = dict(zip(range(number_of_nodes), sorted(nodes)))
    if filepath_pickle is not None:
        filepath_pickle = filepath_pickle + '_ids_to_nodes.p'
        with open(filepath_pickle, 'wb') as file:
            pickle.dump(id2node, file)
    return isnumbers, id2node, number_of_nodes


def read_nodes_from_file(input, sep):
    nodes = set()
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#'):
                continue
            line = line.strip()
            if sep is None:
                splt = line.split()
            else:
                splt = line.split(sep)
            if format == "edgelist":
                if len(splt) == 3:
                    if abs(float(splt[2]) - 1) >= 1e-4:
                        raise ValueError("Weighted graphs are not supported")
                    else:
                        splt = splt[:-1]
                else:
                    raise ValueError("Incorrect graph format")
            for node in splt:
                nodes.add(node)
    return nodes


def build_graph(input, sep, node2id: dict, undirected, isnumbers):
    graph = defaultdict(set)
    with open(input, 'r') as inf:
        for line in inf:
            if line.startswith('#'):
                continue
            line = line.strip()
            if sep is None:
                splt = line.split()
            else:
                splt = line.split(sep)
            if isnumbers:
                src = node2id[int(splt[0])]
            else:
                src = node2id[splt[0]]
            if format == "edgelist" and len(splt) == 3:
                splt = splt[:-1]
            for node in splt[1:]:
                if isnumbers:
                    tgt = node2id[int(node)]
                else:
                    tgt = node2id[node]
                graph[src].add(tgt)
                if undirected:
                    graph[tgt].add(src)
    return graph


def process(format, matfile_variable_name, undirected, sep, input, output, filepath_pickle):
    if format == "mat":
        indptr, indices = mat2xgfs(input, undirected, matfile_variable_name)
    elif format in ['edgelist', 'adjlist']:
        indptr, indices = list2mat(input, undirected, sep, filepath_pickle)

    with open(output, 'wb') as fout:
        xgfs2file(fout, indptr, indices)


@click.command(help=__doc__)
@click.option('--format',
              default='edgelist',
              type=click.Choice(['mat', 'edgelist', 'adjlist']),
              help='File format of input file')
@click.option('--matfile-variable-name', default='network',
              help='variable name of adjacency matrix inside a .mat file.')
@click.option('--undirected/--directed', default=True, is_flag=True,
              help='Treat graph as undirected.')
@click.option('--filepath_pickle', default=None,
              help='file path without file ending, where to write node-to-id + id-to-node mapping dict as pickle file')
@click.option('--sep', default=' ', help='Separator of input file')
@click.argument('input', type=click.Path())
@click.argument('output', type=click.Path())
def main(format, matfile_variable_name, undirected, sep, input, output, filepath_pickle):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)
    logging.info('convert graph from %s to %s and store mapping in {}', input, output, filepath_pickle)
    process(format, matfile_variable_name, undirected, sep, input, output, filepath_pickle)
    logging.info('done.')


if __name__ == "__main__":
    exit(main())
