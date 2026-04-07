import torch
import torch.nn.functional as F


def train_one_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = F.mse_loss(pred[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index)
    loss = F.mse_loss(pred[mask], data.y[mask])
    return float(loss.item())