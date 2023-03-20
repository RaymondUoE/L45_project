import torch
import json, joblib, random
from torch import optim, nn, utils, Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from rnn_data import *
from rmpnn import *

def train_rmpnn(data, model, optimiser, mode='Graph'):
    model.train()
    optimiser.zero_grad()
    counter = 0
    for batch in tqdm(data, desc='training loop'):
        #print(batch)
        try:
            graph_preds, node_preds = model(batch['seq'])

            if mode == 'Graph':
                ys = batch['graph_ys']
                preds = graph_preds
            elif mode == 'Node':
                ys = batch['node_ys']
                preds = node_preds
                
            loss = torch.tensor(0, dtype=torch.float64)
            for t in range(len(graph_preds)): #accumulate loss per time step
                y = ys[t]
                loss += F.cross_entropy(preds[t], y)

            loss = loss/len(graph_preds)
            loss.backward()
            optimiser.step()
        except:
            counter +=1
    print(counter)
    return loss.data

def model_eval(model, test, mode='Graph'):
    total = 0
    correct = 0
    model.eval()
    error = 0
    fn = 0
    tp = 0

    for data in test:
        try:
            with torch.no_grad():
                graph_preds, node_preds = model(data['seq'])
                # Mean Absolute Error using std (computed when preparing data)
                for t in range(len(data['seq'])): # time steps
                    if mode == 'Graph':
                        y_pred = torch.argmax(graph_preds[t], -1).item()
                        y = data['graph_ys'][t].item()
                        if y_pred == y:
                            correct +=1
                        if y_pred == 0 and y != 0:
                            fn += 1
                        if y_pred != 0 and y != 0:
                            tp += 1
                        total +=1
                    elif mode == 'Node':
                        y_pred = torch.argmax(node_preds[t], 1)
                        y = data['node_ys'][t]
                        for i in range(y_pred.shape[0]):
                            if y_pred[i].item() == y[i].item():
                                correct +=1
                            if y_pred[i].item() != 0 and y[i].item() != 0:
                                tp += 1
                            if y_pred[i].item() == 0 and y[i].item() != 0:
                                fn += 1
                            total +=1
        except:
            error += 1
    print(error)
    return correct/total, (fn / (fn + tp))

def update_stats(training_stats, epoch_stats):
    """ Store metrics along the training
    Args:
      epoch_stats: dict containg metrics about one epoch
      training_stats: dict containing lists of metrics along training
    Returns:
      updated training_stats
    """
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key,val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


def train_eval_loop(model, train_seqs, valid_seqs, test_seqs, mode='Node'):
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    training_stats = None
    # Training loop
    for epoch in range(2):
        train_loss = train_rmpnn(train_seqs, model, optimiser, mode=mode)
        train_acc, _ = model_eval(model, train_seqs, mode=mode)
        valid_acc, _ = model_eval(model, valid_seqs, mode=mode)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train accuracy: {train_acc:.3f}",
                    f"validation accuracy: {valid_acc:.3f}")
        # store the loss and the accuracy for the final plot
        epoch_stats = {'train_acc': train_acc, 'val_acc': valid_acc, 'epoch':epoch}
        training_stats = update_stats(training_stats, epoch_stats)
    # Lets look at our final test performance
    test_acc, fnr = model_eval(model, test_seqs, mode=mode)
    print(f"Our final test accuracy for the RMPNN is: {test_acc:.4f}")
    print(f"Our final test FNR for the RMPNN is: {fnr:.4f}")
    return training_stats

if __name__ == '__main__':
    trainLoader, validLoader,testLoader = prepare_data('data/graphs/rnn.json')

    model =  RMPNNModel(num_layers=4, emb_dim=64, in_dim=1, edge_dim=1, graph_out_dim=2, node_out_dim=2)
    
    train_stats_mlp_cora = train_eval_loop(model, trainLoader, validLoader, testLoader, mode='Graph')