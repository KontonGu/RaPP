import pytorch_lightning as pl
import torch_geometric.nn as pyg_nn
import torch.nn as nn
from torchmetrics import MeanAbsolutePercentageError
import torch
from torch.nn import Linear


import torch.nn.functional as F


import math

def init_tensor(tensor, init_type):
    if tensor is None or init_type is None:
        return
    if init_type =='thomas':
        size = tensor.size(-1)
        stdv = 1. / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    else:
        raise ValueError(f'Unknown initialization type: {init_type}')


class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_gat=True):
        super().__init__()
        if use_gat:
            self.conv1 = pyg_nn.GATConv(input_dim, hidden_dim)
            self.conv2 = pyg_nn.GATConv(hidden_dim, hidden_dim)
            self.conv3 = pyg_nn.GATConv(hidden_dim, hidden_dim)
            self.conv4 = pyg_nn.GATConv(hidden_dim, output_dim)
        else:
            self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
            self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
            self.conv3 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
            self.conv4 = pyg_nn.GCNConv(hidden_dim, output_dim)
            

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        return x.mean(dim=0)  # Global pooling (mean over nodes)


class GlobalFeatureProcessor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, hidden_dim)
        self.fc4 = Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class RaPPModel(pl.LightningModule):
    def __init__(self, node_input_dim=40, global_input_dim=12, hidden_dim=512, fc_hidden=512, use_gat=True):
        super().__init__()
        self.gnn = GNNEncoder(node_input_dim, hidden_dim, hidden_dim, use_gat)
        self.global_processor = GlobalFeatureProcessor(global_input_dim, hidden_dim, hidden_dim)
        self.fc_1 = Linear(hidden_dim * 2, fc_hidden)
        self.fc_2 = Linear(fc_hidden, fc_hidden)
        self.fc_3 = Linear(fc_hidden, fc_hidden)
        self.fc_4 = Linear(fc_hidden, fc_hidden)
        self.fc_drop_1 = nn.Dropout(p=0.05)
        self.fc_drop_2 = nn.Dropout(p=0.05)
        self.fc_drop_3 = nn.Dropout(p=0.05)
        self.fc_drop_4 = nn.Dropout(p=0.05)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()
        self.fc_relu3 = nn.ReLU()
        self.fc_relu4 = nn.ReLU()
        
        self.final_fc = nn.Linear(fc_hidden, 256)
        self.predictor = nn.Linear(256, 1)
        
        self._initialize_weights()

        self.train_loss = MeanAbsolutePercentageError()
        self.val_loss = MeanAbsolutePercentageError()
        self.test_loss = MeanAbsolutePercentageError()
        
        
       
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_tensor(m.weight, "thomas")
                init_tensor(m.bias, "thomas") 
        
        
    def forward(self, data):
        node_features, edge_index, global_features = data.x, data.edge_index, data.graph_static.view(1, -1)
        graph_embedding = self.gnn(node_features, edge_index)
        global_embedding = self.global_processor(global_features)
        
        # print('-----------konton_test-------------')
        # print(graph_embedding.unsqueeze(0).shape)
        # print(global_embedding)
        x = torch.cat([graph_embedding.unsqueeze(0), global_embedding], dim=-1)
        x = self.fc_1(x)
        x = self.fc_relu1(x)
        x = self.fc_drop_1(x)
        x = self.fc_2(x)
        x = self.fc_relu2(x)
        x = self.fc_drop_2(x)
        x = self.fc_3(x)
        x = self.fc_relu3(x)
        x = self.fc_drop_3(x)
        x = self.fc_4(x)
        x = self.fc_relu4(x)
        feat = self.fc_drop_4(x)
        feat = self.final_fc(feat)
        x = self.predictor(feat)
        
        pred = -F.logsigmoid(x)
        return pred

    
    
    def training_step(self, data, batch_idx):
        data.y = torch.Tensor([[data.y]])
        data = data.to(device=torch.device("cuda"))
        y_hat = self(data)
        y = data.y
        loss = F.huber_loss(y_hat, y)
        self.train_loss(y_hat, y)
        self.log('train_loss', self.train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        data.y = torch.Tensor([[data.y]])
        data = data.to(device=torch.device("cuda"))
        y_hat = self(data)    
        y = data.y
        self.val_loss(y_hat, y)
        self.log('val_loss', self.val_loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, data, batch_idx):
        data.y = torch.Tensor([[data.y]])
        data = data.to(device=torch.device("cuda"))
        y_hat = self(data)
        y = data.y
        self.test_loss(y_hat, y)
        self.log('test_loss', self.test_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2.7542287033381663e-05)