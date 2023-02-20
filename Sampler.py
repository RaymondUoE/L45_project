import argparse
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import json
import random
import pickle

# IN_FILES = []
# OUT_FILE = ''
# NUM_SAMPLE = 0

def sample_graphs(G, sample_space, hop=5, cut_off=30):
    out_graphs = []
    print(f'Sampling {NUM_REAL} real life graphs...')
    for i in range(NUM_REAL):
        sample_node = np.random.choice(sample_space, 1)[0]
        nodes = [sample_node]
        frontier = [sample_node]
        visited = [sample_node]
        for h in range(hop):
            new_frontier = []
            for f in frontier:
                neighbours = [x for x in list(G.adj[f]) if x not in visited]
                if len(neighbours) == 0:
                    continue
                elif len(neighbours) > cut_off:
                    neighbours = random.sample(neighbours, cut_off)
                nodes += neighbours
                new_frontier += neighbours
                visited.append(f)
            frontier = new_frontier
            
        out_graphs.append(nx.convert_node_labels_to_integers(G.subgraph(nodes)))
    return out_graphs

def store_as_list_of_dicts(filename, graphs):

    list_of_dicts = [nx.to_dict_of_dicts(graph) for graph in graphs]

    with open(filename, 'wb') as f:
        pickle.dump(list_of_dicts, f)
        f.close()
    return

def load_list_of_dicts(filename, create_using=nx.Graph):
    
    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)
    
    graphs = [create_using(graph) for graph in list_of_dicts]
    
    return graphs

if __name__ == '__main__':
    
    with open('configs/sampler_config.json', 'r') as f:
        config = json.load(f)
        f.close()
    
    IN_FILES = config['in_file_paths']
    OUT_FILE = config['out_file_path']
    NUM_SAMPLE = config['num_of_samples']
    REAL_ALPHA = config['real_data_alpha']
    NUM_REAL = int(NUM_SAMPLE * REAL_ALPHA)
    NUM_BA = NUM_SAMPLE - NUM_REAL
    
    data = pd.read_csv(IN_FILES[0], index_col=0)
    for file in IN_FILES[1:]:
        
        data.append(pd.read_csv(file, index_col=0), ignore_index=True)
    
    unique_originh = list(set(data['originh']))
    unique_responh = list(set(data['responh']))
    unique_all = list(set(unique_originh + unique_responh))

        
    print('Constructing full graph')
    G = nx.Graph()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        G.add_edge(row['originh'], row['responh'])

    hop = 5
    sampled_graphs = sample_graphs(G, unique_originh, hop=hop)
    
    print(f'Generating {NUM_BA} Barabasi Albert graphs...')
    for i in range(NUM_BA):
        ba_graph = nx.barabasi_albert_graph(20,2)
        sampled_graphs.append(ba_graph)
        
    store_as_list_of_dicts(OUT_FILE, sampled_graphs)
    

