import json, joblib, random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Data

class GraphSeq(Dataset):   
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = {'seq':[], 'graph_ys':[], 'node_ys':[]}
        for t in range(len(self.data[index])):
            x['seq'].append(self.data[index][t])
            x['graph_ys'].append(self.data[index][t].graph_y)
            x['node_ys'].append(self.data[index][t].y)

        return x
    
def json2graphs(data):
    
    sequences = []
    for i in range(len(data['graphs'])):
        graph_sequence = []
        for t in range(len(data['graphs'][0])): #t time steps
            adj = torch.tensor(data['graphs'][i][t])
            edge_index = adj.nonzero().t().contiguous()
            edge_attr = []
            edge_labels = []
            node_labels = []
            for j in range(edge_index.shape[1]):
                source = edge_index[0][j].item()
                target = edge_index[1][j].item()
                edge_attr.append(data['clusters'][i][t][source][target])
                edge_labels.append(data['labels'][i][t][source][target])
            
            node_labels = [0 for i in range(len(data['graphs'][i][t]))]
            graph_label = 0
            for idx in range(len(edge_labels)):
                if edge_labels[idx] > 0:
                    source = edge_index[0][idx]
                    node_labels[int(source)] = 1
                    graph_label = 1

            x = torch.ones((len(data['graphs'][i][t]), 1), dtype=torch.float)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).reshape((edge_index.shape[1], 1))

            node_labels = torch.tensor(node_labels).reshape(x.shape)
            edge_labels = torch.tensor(edge_labels).reshape((edge_attr.shape[0], 1))
            graph_label = torch.tensor(graph_label)
            pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=node_labels, graph_y=graph_label, edge_y=edge_labels)
            graph_sequence.append(pyg_graph)

        sequences.append(graph_sequence)
    return sequences


def prepare_data(file, shuffle=False):
    with open(file, 'r') as json_file:
        data = json.load(json_file)

    sequences = json2graphs(data)
    if shuffle:
        random.shuffle(sequences)

    train = GraphSeq(sequences[:int(0.8*len(sequences))])
    valid = GraphSeq(sequences[int(0.8*len(sequences)):int(0.9*len(sequences))])
    test =  GraphSeq(sequences[int(0.9*len(sequences)):])

    # train_loader = DataLoader(train)
    # valid_loader = DataLoader(valid)
    # test_loader = DataLoader(test)
    # train = sequences[:int(0.8*len(sequences))]
    # valid = sequences[int(0.8*len(sequences)):int(0.9*len(sequences))]
    # test =  sequences[int(0.9*len(sequences)):]
    train_loader = train
    valid_loader = valid
    test_loader = test

    return train_loader, valid_loader, test_loader