from engine import value
import numpy as np


def binary_crossentropy(y_true, y_pred):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    loss = - (y_true * y_pred.log() + (1. - y_true) * (1. - y_pred).log())
    loss = loss.mean()
    return loss

def categorical_crossentropy(y_true, y_pred):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    loss = - (y_true * y_pred.log()).sum(axis=-1)
    loss = loss.mean()
    return loss

def sparse_categorical_crossentropy(y_true, y_pred):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    y_true_one_hot = value(np.eye(y_pred.shape[1])[y_true.data])  # Convert to one-hot encoding
    loss = - (y_true_one_hot * y_pred.log()).sum(axis=-1)
    loss = loss.mean()
    return loss

def poisson(y_true, y_pred):
    loss = y_pred - y_true * (y_pred + value(1e-12)).log()
    loss = loss.mean()
    return loss

def kl_divergence(p, q):
    kl_div = (p * (p / q).log()).sum(axis=-1)
    loss = kl_div.mean()
    return loss

def mean_squared_error(y_true, y_pred):
    diff = y_pred - y_true
    diff2 = diff ** 2
    loss = diff2.mean()
    return loss

def mean_absolute_error(y_true, y_pred):
    diff = (y_pred - y_true).abs()
    loss = diff.mean()
    return loss

def mean_absolute_percentage_error(y_true, y_pred):
    diff = ((y_true - y_pred) / y_true.abs().clip(value(1e-12))).abs()
    loss = diff.mean() * 100
    return loss

def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = (y_pred + value(1)).log()
    second_log = (y_true + value(1)).log()
    loss = ((first_log - second_log) ** 2).mean()
    return loss

def cosine_similarity(y_true, y_pred):
    y_true_norm = y_true / (y_true ** 2).sum(axis=-1, keepdims=True).sqrt()
    y_pred_norm = y_pred / (y_pred ** 2).sum(axis=-1, keepdims=True).sqrt()
    similarity = (y_true_norm * y_pred_norm).sum(axis=-1)
    return similarity.mean()

def huber(y_true, y_pred, delta=1.0):
    diff = y_true - y_pred
    condition = (diff.abs() <= delta).data
    squared_loss = value(0.5) * (diff ** 2)
    linear_loss = delta * (diff.abs() - value(0.5) * delta)
    loss = value(np.where(condition, squared_loss.data, linear_loss.data)).mean()
    return loss

def log_cosh(y_true, y_pred):
    loss = (y_pred - y_true).cosh().log()
    return loss.mean()

def hinge(y_true, y_pred):
    loss = (value(1.) - y_true * y_pred).clip(value(0.), None)
    return loss.mean()

def squared_hinge(y_true, y_pred):
    loss = (value(1.) - y_true * y_pred).clip(value(0.), None) ** 2
    return loss.mean()

def categorical_hinge(y_true, y_pred):
    pos = (y_true * y_pred).sum(axis=-1)
    neg = ((value(1.) - y_true) * y_pred).max(axis=-1)
    loss = (value(1.) + neg - pos).clip(value(0.), None)
    return loss.mean()

def binary_focal_crossentropy(y_true, y_pred, gamma=2., alpha=0.25):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    loss = - alpha * (1 - y_pred) ** gamma * y_true * y_pred.log() - (1 - alpha) * y_pred ** gamma * (1 - y_true) * (1 - y_pred).log()
    return loss.mean()

def categorical_focal_crossentropy(y_true, y_pred, gamma=2., alpha=0.25):
    eps = value(1e-12)  # to prevent log(0)
    y_pred = y_pred.clip(eps, 1. - eps)
    loss = - y_true * ((1 - y_pred) ** gamma) * y_pred.log() * alpha
    return loss.mean(axis=-1).mean()

