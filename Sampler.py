import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import json
import random
import pickle

def sample_graphs(G, sample_space, hop=3, cut_off=20):
    '''Sample a subgraph from a large graph G'''
    
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
    '''convert a string valued flow duration feature to float in seconds'''
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
    '''sample n edge features from malicious dataframe'''
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

def create_dynamic_networks(adj_mat, drop_prob, add_prob, t):
    '''create a series of dynamic networks over time t'''
    if t == 0:
        return []
    else:
        nnode = len(adj_mat)
        drop_edges = np.floor(np.random.rand(nnode,nnode) / (1-drop_prob))
        new_adj = ((adj_mat - drop_edges) > 0).astype(int)
        add_edges = np.floor(np.random.rand(nnode,nnode) / (1-add_prob))
        new_adj = ((new_adj + add_edges) > 0).astype(int)
        return [new_adj] + create_dynamic_networks(new_adj, drop_prob, add_prob, t-1)

def assign_clusters(number_of_types, size):
    '''assign dummy cluster/traffic type for each edge'''
    return np.random.randint(0, number_of_types, size=size)  

def assign_labels(clusters, attack_combinations):
    '''assign labels based on config'''
    
    nnode = clusters.shape[1]
    labels = []
    for t, cls_t in enumerate(clusters):
        history = clusters[:t+1, :, :]
        history = np.transpose(history, axes=(1,2,0))
        
        labels_t = np.zeros((nnode, nnode))
        for atk in attack_combinations:
            if len(atk) > history.shape[-1]:
                continue
            
            repeated_perm = np.tile(atk, [nnode, nnode]).reshape(nnode, nnode, -1)
            labels_t += (history[:,:,-len(atk):]==repeated_perm).all(axis=2).astype(int)
        labels_t[cls_t == -1] = -1
        labels.append(labels_t)
    
    return np.stack(labels)

def store_dict(save_path, ddict):
    with open(save_path, 'w') as f:
        json.dump(ddict, f)
        f.close()

if __name__ == '__main__':
    
    with open('configs/sampler_config.json', 'r') as f:
        config = json.load(f)
        f.close()
    
    IN_FILES = config['in_file_paths']
    NUM_SAMPLE = config['num_of_samples']
    REAL_ALPHA = config['real_data_alpha'] # ratio of subsampled graphs v.s. babarasi graphs
    NUM_REAL = int(NUM_SAMPLE * REAL_ALPHA)
    NUM_BA = NUM_SAMPLE - NUM_REAL
    IS_RNN = config['sample_rnn']
    
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
    # bf_df = data[data['attack_category']=='Bruteforce']
    # bfxml_df = data[data['attack_category']=='Bruteforce-XML']
    # crypto_df = data[data['attack_category']=='XMRIGCC CryptoMiner']
    malicious_df = data[(data['attack_category']!='Benign') & (data['attack_category']!='Background')]
    
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
        
    
    if IS_RNN:
        OUT_FILE = config['out_rnn_path']
        with open('configs/recurrent_sampler_config.json', 'r') as f:
            rnn_config =  json.load(f)
            f.close()
        
        graphs_all, clusters_all, labels_all = [], [], []
        
        for graph in tqdm(sampled_graphs, total=len(sampled_graphs)):
            dg = nx.DiGraph()
            dg.add_edges_from(list(graph.edges))
            adj_mat = nx.to_numpy_array(dg)

            dynamic_networks = np.stack(create_dynamic_networks(adj_mat, rnn_config['add_edge_prob'], rnn_config['drop_edge_prob'], rnn_config['t']))
            clusters = []
            for g_t in dynamic_networks:
                clusters.append(np.select([g_t == 0, g_t !=0], [-1, assign_clusters(rnn_config['number_of_types'], size=(len(g_t),len(g_t)))]))

            clusters = np.stack(clusters)

            labels = assign_labels(clusters, rnn_config['attack_combinations'])
            
            graphs_all.append(dynamic_networks)
            clusters_all.append(clusters)
            labels_all.append(labels)
            
        out_dict = {}
        out_dict['graphs'] = [x.tolist() for x in graphs_all]
        out_dict['clusters'] = [x.tolist() for x in clusters_all]
        out_dict['labels'] = [x.tolist() for x in labels_all]
        
        store_dict(OUT_FILE, out_dict)
        
        
    else:
        OUT_FILE = config['out_file_path']
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
    

