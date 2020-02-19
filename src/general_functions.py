import numpy as np


def l2_unit_ball_projection(vector):
    return vector / np.maximum(1, np.linalg.norm(vector, ord=2))


def subgradient(x, y, w, loss_name=None):
    if loss_name == 'hinge':
        score = x @ w
        if y * score < 1:
            return -y
        else:
            return 0
    elif loss_name == 'absolute':
        pred = x @ w
        if y < pred:
            return 1
        elif y >= pred:
            return -1
    else:
        raise ValueError("Unknown loss.")


def loss(x, y, w, loss_name='hinge'):
    if loss_name == 'hinge':
        from sklearn.metrics import hinge_loss
        loss_thing = hinge_loss(y, x @ w)
        return loss_thing
    elif loss_name == 'absolute':
        if hasattr(y, '__len__'):
            return 1 / len(y) * np.sum(np.abs(y - x @ w))
        else:
            return 1 / len(y) * np.sum(np.abs(y - x @ w))
    else:
        raise ValueError("Unknown loss.")
