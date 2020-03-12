import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def l2_unit_ball_projection(vector):
    return vector / np.maximum(1, np.linalg.norm(vector, ord=2))


def subgradient(x, y, w, loss_name=None):
    if loss_name == 'hinge':
        pred_scores = x @ w
        n_classes = w.shape[1]

        indicator_part = np.ones(pred_scores.shape)
        indicator_part[y] = 0

        true = pred_scores[y]
        true = np.tile(true, (1, n_classes))

        j_star = np.argmax(indicator_part + pred_scores - true)

        subgrad = np.zeros(w.shape)

        if y != j_star:
            subgrad[:, j_star] = x
            subgrad[:, y] = -x
        return subgrad
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
        def multiclass_hinge_loss(curr_labels, pred_scores):
            indicator_part = np.ones(pred_scores.shape)
            indicator_part[np.arange(pred_scores.shape[0]), curr_labels] = 0

            true = pred_scores[np.arange(pred_scores.shape[0]), curr_labels].reshape(-1, 1)
            true = np.tile(true, (1, 4))

            loss = np.max(indicator_part + pred_scores - true, axis=1)
            loss = np.sum(loss) / len(curr_labels)
            return loss
        if hasattr(y, '__len__'):
            return multiclass_hinge_loss(y, x @ w)
        else:
            scores = x @ w
            n_classes = w.shape[1]
            indicator_part = np.ones(scores.shape)
            indicator_part[y] = 0

            true = scores[y]
            true = np.tile(true, (1, n_classes))

            return np.max(indicator_part + scores - true, axis=1)[0]

    elif loss_name == 'absolute':
        if hasattr(y, '__len__'):
            return 1 / len(y) * np.sum(np.abs(y - x @ w))
        else:
            return np.abs(y - x @ w)
    else:
        raise ValueError("Unknown loss.")


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        fig_manager = matplotlib._pylab_helpers.Gcf.get_active()
        if fig_manager is not None:
            canvas = fig_manager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return
