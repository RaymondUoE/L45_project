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

def sample_graphs(G, sample_space, hop=3, cut_off=20):
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

    # with open(filename, 'wb') as f:
    #     pickle.dump(list_of_dicts, f)
    #     f.close()
    with open(filename, 'w') as f:
        for item in list_of_dicts:
            f.write(json.dumps(item) + "\n")
        f.close()
    return

def load_list_of_dicts(filename, create_using=nx.Graph):
    
    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)
    
    graphs = [create_using(graph) for graph in list_of_dicts]
    
    return graphs

def flow_duration_to_float(flow):
    # checking if its string
    if 'days' in str(flow):
        days = float(flow.split('days', 1)[0].strip())
        hh = days * 24
        remain = flow.split('days', 1)[1]
        hh += float(remain.split(':', 1)[0].strip())
        mm = hh * 60
        remain = remain.split(':', 1)[1]
        mm += float(remain.split(':', 1)[0].strip())
        remain = remain.split(':', 1)[1]
        ss = mm * 60
        ss += float(remain)
        return ss
    else:
        return float(flow)

def sample_edge_attributes(num_of_edges, benign_df, malicious_df, pos_prob, features):
    attributes = []
    for i in range(num_of_edges):
        # sample from malicious
        if np.random.uniform(0,1) < pos_prob:
            sample = malicious_df.sample(1)
            feature_values = sample[features].to_dict('records')[0]
            # add label
            feature_values['label'] = categorial_to_id[sample['attack_category'].values[0]]
            attributes.append(feature_values)
        else:
            sample = benign_df.sample(1)
            feature_values = sample[features].to_dict('records')[0]
            # add label
            feature_values['label'] = categorial_to_id[sample['attack_category'].values[0]]
            attributes.append(feature_values)
    return attributes

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
    
    dfs = []
    for file in IN_FILES:
        df = pd.read_csv(file, index_col=0)
        if 'traffic_category' in df.columns:
            df = df.rename(columns={"traffic_category": "attack_category"})
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    
    data['flow_duration'] = data['flow_duration'].apply(lambda x: flow_duration_to_float(x))
    benign_df = data[data['attack_category']=='Benign']
    bf_df = data[data['attack_category']=='Bruteforce']
    bfxml_df = data[data['attack_category']=='Bruteforce-XML']
    crypto_df = data[data['attack_category']=='XMRIGCC CryptoMiner']
    malicious_df = data[data['attack_category']!='Benign']
    
    categorial_to_id = {
        'Benign': 0,
        'Background': 0,
        'Bruteforce': 1,
        'Bruteforce-XML': 2,
        'Probing': 3,
        'XMRIGCC CryptoMiner': 4
    }
    
    unique_originh = list(set(data['originh']))
    unique_responh = list(set(data['responh']))
    unique_all = list(set(unique_originh + unique_responh))

        
    print('Constructing full graph')
    G = nx.Graph()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        G.add_edge(row['originh'], row['responh'])

    hop = config['num_of_hops']
    sampled_graphs = sample_graphs(G, unique_originh, hop=hop)
    
    print(f'Generating {NUM_BA} Barabasi Albert graphs...')
    for i in range(NUM_BA):
        ba_graph = nx.barabasi_albert_graph(20,2)
        sampled_graphs.append(ba_graph)
    
    for graph in tqdm(sampled_graphs, total=len(sampled_graphs)):
        # graph might have positive labels
        if np.random.uniform(0,1) < config['force_pos_graph_ratio']:
            edge_attrs = sample_edge_attributes(len(graph.edges), benign_df, malicious_df, config['force_pos_lable_prob'], config['features'])
        # graph has all negative labels
        else:
            edge_attrs = sample_edge_attributes(len(graph.edges), benign_df, malicious_df, 0, config['features'])
        edge_features = {}
        for i, k in enumerate(graph.edges):
            edge_features[k] = edge_attrs[i]
        nx.set_edge_attributes(graph, values=edge_features)
    
    store_as_list_of_dicts(OUT_FILE, sampled_graphs)
    

