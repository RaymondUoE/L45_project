
import torch
import json, joblib, random
from torch import optim, nn, utils, Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from prepare_data import *
from mpnn import *


class GraphMPNN(pl.LightningModule):
    def __init__(self, emb_dim, in_dim, edge_dim, graph_out_dim=1, node_out_dim=5, mode='Graph', load=False, load_path=None):
        super().__init__()
        self.mode = mode
        self.model = MPNNModel(num_layers=4, emb_dim=emb_dim, in_dim=in_dim, edge_dim=edge_dim, graph_out_dim=graph_out_dim, node_out_dim=node_out_dim)
        if load:
            self.model.load_state_dict(torch.load(load_path))
        
    def forward(self, batch):
        graph_pred, node_pred= self.model(batch)
        if self.mode == 'Graph':
            y = batch.graph_y
            pred = graph_pred
        elif self.mode == 'Node':
            y = batch.y.squeeze(1)
            pred = node_pred
        #print(y.shape)
        #print(pred.shape)

        loss = F.cross_entropy(pred, y)

        return loss, pred
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        
        # Logging to TensorBoard by default
        loss, _ = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
def trainMPNN(train_dataloader, val_dataloader, epochs, batch_size, emb_dim, in_dim, edge_dim, graph_out_dim=1, node_out_dim=5, mode='Graph', load=False, load_path=None):
    model = GraphMPNN(emb_dim, in_dim, edge_dim, graph_out_dim=graph_out_dim, node_out_dim=node_out_dim, mode=mode, load=False, load_path=None)

    save_path = os.path.join('GraphMPNN_', '_'.join([str('epochs')]))
    trainer = pl.Trainer(limit_train_batches=batch_size, max_epochs=epochs, accelerator="cuda", callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
   
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path+'/model.pt')

    return model

def model_eval(model, test, mode='Graph'):
    total = 0
    correct = 0
    model.eval()
    model.to('cuda')
    error = 0

    for data in test:
        data = data.to('cuda')
        with torch.no_grad():
            _, y_pred = model(data)
            # Mean Absolute Error using std (computed when preparing data)
            if mode == 'Graph':
                y_pred = torch.argmax(y_pred, -1).item()
                y = data.graph_y
                if y_pred == data.graph_y.item():
                    correct +=1
                total +=1
            elif mode == 'Node':
                y_pred = torch.argmax(y_pred, 1)
                y = data.y
                for i in range(y_pred.shape[0]):
                    if y_pred[i].item() == y[i].item():
                        correct +=1
                    total +=1
    
    return correct/total


if __name__ == '__main__':
    in_dim, edge_dim, trainLoader, validLoader,testLoader = prepare_data('data/graphs/graphs.jsonl')

    model = trainMPNN(trainLoader, validLoader, 50, 128, 128, in_dim, edge_dim, 1, 5, 'Node')
    
    print('TESTING......')
    acc = model_eval(model, testLoader, mode='Node')
    print(acc)


