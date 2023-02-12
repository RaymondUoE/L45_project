import argparse
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_name', type=str, help='input raw file name in csv', required=True)
parser.add_argument('-n', '--number', type=int, help='number of graphs sampled', required=True)
parser.add_argument('-o', '--out_file', type=str, help='output file', required=True)
args = parser.parse_args()
print(args.file_name)
file_name = args.file_name
num_of_sample = args.number
out_file = args.out_file
data = pd.read_csv(file_name, index_col=0)

unique_originh = list(set(data['originh']))
unique_responh = list(set(data['responh']))
unique_all = list(set(unique_originh + unique_responh))

    
G = nx.Graph()
for i, row in tqdm(data.iterrows(), total=len(data)):
    G.add_edge(row['originh'], row['responh'])

sample_node = np.random.choice(unique_originh, 1)[0]

hop = 5
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
    
subgraph = G.subgraph(nodes)
        