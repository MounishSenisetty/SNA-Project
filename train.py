import torch
import torch.nn.functional as F
from time import perf_counter


def train_one_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = F.mse_loss(pred[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_one_epoch_adv(model, data, optimizer, train_mask, grad_clip=None):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index)
    loss = F.mse_loss(pred[train_mask], data.y[train_mask])
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index)
    loss = F.mse_loss(pred[mask], data.y[mask])
    return float(loss.item())


def fit_model(
    model,
    data,
    train_mask,
    val_mask,
    optimizer,
    epochs: int,
    patience: int = 30,
    scheduler=None,
    grad_clip=None,
    verbose_every: int = 20,
):
    best_val = float("inf")
    best_state = None
    wait = 0
    history = []
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch_adv(model, data, optimizer, train_mask, grad_clip=grad_clip)
        va = eval_loss(model, data, val_mask)
        if scheduler is not None:
            scheduler.step(va)

        if va < best_val:
            best_val = va
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        history.append({"epoch": epoch, "train_mse": tr, "val_mse": va})
        if epoch % verbose_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train MSE {tr:.4f} | val MSE {va:.4f}")
        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    elapsed = perf_counter() - start
    return {
        "best_val_mse": float(best_val),
        "epochs_ran": len(history),
        "train_seconds": float(elapsed),
        "history": history,
    }