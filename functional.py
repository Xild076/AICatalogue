from engine import value
import numpy as np

def binary_crossentropy(y_pred, y_true, reduction='mean'):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    loss = - (y_true * y_pred.log() + (1. - y_true) * (1. - y_pred).log())
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

def categorical_crossentropy(y_pred, y_true, reduction='mean'):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    loss = - (y_true * y_pred.log()).sum(axis=-1)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

def sparse_categorical_crossentropy(y_pred, y_true, reduction='mean'):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    y_true_one_hot = value(np.eye(y_pred.shape[1])[y_true.data])  # Convert to one-hot encoding
    loss = - (y_true_one_hot * y_pred.log()).sum(axis=-1)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

def poisson(y_pred, y_true, reduction='mean'):
    loss = y_pred - y_true * (y_pred + value(1e-12)).log()
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

def kl_divergence(q, p, reduction='mean'):
    kl_div = (q * (q / p).log()).sum(axis=-1)
    if reduction == 'mean':
        loss = kl_div.mean()
    elif reduction == 'sum':
        loss = kl_div.sum()
    return loss

def mean_squared_error(y_pred, y_true, reduction='mean'):
    diff = y_pred - y_true
    diff2 = diff ** 2
    if reduction == 'mean':
        loss = diff2.mean()
    elif reduction == 'sum':
        loss = diff2.sum()
    elif reduction == 'none':
        loss = diff2
    return loss

def mean_absolute_error(y_pred, y_true, reduction='mean'):
    diff = (y_pred - y_true).abs()
    if reduction == 'mean':
        loss = diff.mean()
    elif reduction == 'sum':
        loss = diff.sum()
    elif reduction == 'none':
        loss = diff
    return loss

def mean_absolute_percentage_error(y_pred, y_true, reduction='mean'):
    diff = ((y_true - y_pred) / y_true.abs().clip(value(1e-12))).abs()
    loss = diff * 100
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        loss = loss
    return loss

def mean_squared_logarithmic_error(y_pred, y_true, reduction='mean'):
    first_log = (y_pred + value(1)).log()
    second_log = (y_true + value(1)).log()
    diff = (first_log - second_log) ** 2
    if reduction == 'mean':
        loss = diff.mean()
    elif reduction == 'sum':
        loss = diff.sum()
    elif reduction == 'none':
        loss = diff
    return loss

def cosine_similarity(y_pred, y_true, reduction='mean'):
    y_true_norm = y_true / (y_true ** 2).sum(axis=-1, keepdims=True).sqrt()
    y_pred_norm = y_pred / (y_pred ** 2).sum(axis=-1, keepdims=True).sqrt()
    similarity = (y_true_norm * y_pred_norm).sum(axis=-1)
    if reduction == 'mean':
        loss = similarity.mean()
    elif reduction == 'sum':
        loss = similarity.sum()
    elif reduction == 'none':
        loss = similarity
    return loss

def huber(y_pred, y_true, delta=1.0, reduction='mean'):
    diff = y_true - y_pred
    condition = (diff.abs() <= delta).data
    squared_loss = value(0.5) * (diff ** 2)
    linear_loss = delta * (diff.abs() - value(0.5) * delta)
    loss = value(np.where(condition, squared_loss.data, linear_loss.data))
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        loss = loss
    return loss

def log_cosh(y_pred, y_true, reduction='mean'):
    loss = (y_pred - y_true).cosh().log()
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        loss = loss
    return loss

def hinge(y_pred, y_true, reduction='mean'):
    loss = (value(1.) - y_true * y_pred).clip(value(0.), None)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        loss = loss
    return loss

def squared_hinge(y_pred, y_true, reduction='mean'):
    loss = (value(1.) - y_true * y_pred).clip(value(0.), None) ** 2
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        loss = loss
    return loss

def categorical_hinge(y_pred, y_true, reduction='mean'):
    pos = (y_true * y_pred).sum(axis=-1)
    neg = ((value(1.) - y_true) * y_pred).max(axis=-1)
    loss = (value(1.) + neg - pos).clip(value(0.), None)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        loss = loss
    return loss

def binary_focal_crossentropy(y_pred, y_true, gamma=2., alpha=0.25, reduction='mean'):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    loss = - alpha * (1 - y_pred) ** gamma * y_true * y_pred.log() - (1 - alpha) * y_pred ** gamma * (1 - y_true) * (1 - y_pred).log()
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        loss = loss
    return loss

def categorical_focal_crossentropy(y_pred, y_true, gamma=2., alpha=0.25, reduction='mean'):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    loss = - y_true * ((1 - y_pred) ** gamma) * y_pred.log() * alpha
    if reduction == 'mean':
        loss = loss.mean(axis=-1).mean()
    elif reduction == 'sum':
        loss = loss.sum(axis=-1).sum()
    elif reduction == 'none':
        loss = loss.mean(axis=-1)
    return loss