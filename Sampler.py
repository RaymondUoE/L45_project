import argparse
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import json

# IN_FILES = []
# OUT_FILE = ''
# NUM_SAMPLE = 0

def sample_graphs(G, sample_space, hop=5):
    out_graphs = []
    for i in range(NUM_SAMPLE):
        sample_node = np.random.choice(sample_space, 1)[0]
        nodes = [sample_node]
        frontier = [sample_node]
        visited = [sample_node]
        for h in range(hop):
            new_frontier = []
            for f in frontier:
                neighbours = [x for x in list(G.adj[f]) if x not in visited][:5]
                if len(neighbours) == 0:
                    continue
                nodes += neighbours
                new_frontier += neighbours
                visited.append(f)
            frontier = new_frontier
            
        out_graphs.append(G.subgraph(nodes))
    return out_graphs

if __name__ == '__main__':
    
    with open('configs/sampler_config.json', 'r') as f:
        config = json.load(f)
        f.close()
    
    IN_FILES = config['in_file_paths']
    OUT_FILE = config['out_file_path']
    NUM_SAMPLE = config['num_of_samples']
    
    
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

