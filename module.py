import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score
class Predictor(nn.Module):
    def __init__(self, evidence_size=5, layers=[200, 100, 1]):
        # tile scoring
        super().__init__()
        self.conv1d_layer = torch.nn.Conv1d(2048, 1, 1)
        self.evidence_size=evidence_size #R
        linear_layer_list = []
        for i in range(len(layers)):
            if i==0:
                linear_layer_list.append(nn.Linear(2 * self.evidence_size, layers[i]))
            else:
                linear_layer_list.append(nn.Linear(layers[i-1], layers[i]))
            if (i+1)!=len(layers):
                linear_layer_list.append(nn.ReLU())
        self.linear_layers = nn.Sequential(*linear_layer_list)
        self.sigmoid_f = torch.nn.Sigmoid()
    def forward(self, image_features):
        # input [batch, 2048, Tile_size]
        tile_descriptor = self.conv1d_layer(image_features)
        # [batch, 1, Tile_size]
        top_items, _ = torch.topk(tile_descriptor, self.evidence_size, dim=2) 
        bottom_items, _ = torch.topk(tile_descriptor, self.evidence_size, dim=2, largest=False)
        evidences = torch.cat((top_items, bottom_items), dim=2).view(-1, self.evidence_size * 2)
        # items [batch, 2 * evidence_size]
        logits = self.linear_layers(evidences) 
        results = self.sigmoid_f(logits).view(-1)
        return results

def accuracy(result, target):
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return torch.eq(result > 0.5, target).sum().numpy() / len(target)

def auc(result, target):
    if result.is_cuda:
        result = result.cpu()
        target = target.cpu()
    return roc_auc_score(target.numpy(), result.detach().numpy())