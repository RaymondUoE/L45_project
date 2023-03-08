import json, joblib, random
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader

def json2graph(json):
    nodes = set()
    edge_indices = [[], []]
    edge_feats = []
    edge_labels = []

    for source in json.keys():
        for target in json[source].keys():
            edge_indices[0].append(int(source))
            edge_indices[1].append(int(target))
            nodes.add(source)
            nodes.add(target)

            dict_keys = sorted([key for key in json[source][target].keys()])
            dict_keys.remove('label')
            edge_feat = [json[source][target][key] for key in dict_keys]
            edge_feats.append(edge_feat)

            label = json[source][target]['label']
            edge_labels.append(label)

    node_labels = [0 for i in range(len(nodes))]
    graph_label = 0
    for idx in range(len(edge_labels)):
        if edge_labels[idx] > 0:
            source = edge_indices[0][idx]
            node_labels[int(source)] = 1
            graph_label = 1
    
    x = torch.ones((len(nodes), 1), dtype=torch.float)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long)
    edge_feats = torch.tensor(edge_feats, dtype=torch.float)

    node_labels = torch.tensor(node_labels).reshape(x.shape)
    edge_labels = torch.tensor(edge_labels).reshape((edge_feats.shape[0], 1))
    graph_label = torch.tensor(graph_label)

    return Data(x=x, edge_index=edge_indices, edge_attr=edge_feats, y=node_labels, graph_y=graph_label, edge_y=edge_labels)

def prepare_data(file, shuffle=False):
    with open(file, 'r') as json_file:
        json_list = list(json_file)
 
    graphs = []
    for l in json_list:
        graphs.append(json2graph(json.loads(l)))

    if shuffle:
        random.shuffle(graphs)

    train = graphs[:int(0.8*len(graphs))]
    valid = graphs[int(0.8*len(graphs)):int(0.9*len(graphs))]
    test = graphs[int(0.9*len(graphs)):]

    train_loader = DataLoader(train)
    valid_loader = DataLoader(valid)
    test_loader = DataLoader(test)

    return graphs[0].x.shape[1], graphs[0].edge_attr.shape[1], train_loader, valid_loader, test_loader


