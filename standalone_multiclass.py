import numpy as np
# from src.general_functions import l2_unit_ball_projection, subgradient, loss
import matplotlib.pyplot as plt
import pandas as pd
from src.general_functions import mypause
from sklearn.datasets import make_blobs
# import warnings
# warnings.filterwarnings("error")


# parameters
n_points = 1000
dims = 2
signal_to_noise_ratio = 10
n_classes = 4

seed = 9999
np.random.seed(seed)

features, labels = make_blobs(n_samples=n_points, centers=n_classes, n_features=dims, cluster_std=0.2)

plt.scatter(features[np.where(labels == 0), 0], features[np.where(labels == 0), 1])
plt.scatter(features[np.where(labels == 1), 0], features[np.where(labels == 1), 1])
plt.scatter(features[np.where(labels == 2), 0], features[np.where(labels == 2), 1])
plt.scatter(features[np.where(labels == 3), 0], features[np.where(labels == 3), 1])
plt.pause(0.001)


my_dpi = 100
plt.figure(figsize=(800 / my_dpi, 500 / my_dpi), facecolor='white', dpi=my_dpi)
plt.title('blue: parameter-free | red: SGD')
plt.pause(0.001)


def loss(x, y, w):
    pred_scores = x @ w
    indicator_part = np.ones(pred_scores.shape)
    indicator_part[y] = 0

    true = pred_scores[y]
    true = np.tile(true, w.shape[1])

    return np.max(indicator_part + pred_scores - true)


def subgradient(x, y, w):
    pred_scores = x @ w

    indicator_part = np.ones(pred_scores.shape)
    indicator_part[y] = 0

    true = pred_scores[y]
    true = np.tile(true, (1, w.shape[1]))

    j_star = np.argmax(indicator_part + pred_scores - true)

    subgrad = np.zeros(w.shape)
    if y != j_star:
        subgrad[:, j_star] = x
        subgrad[:, y] = -x
    return subgrad


def l2_unit_ball_projection(mat):
    return mat / np.linalg.norm(mat, ord='fro')

################################################################################################
################################################################################################
################################################################################################


# optimization for parameter free
L = np.sqrt(2)
R = 1

# initialize the inner parameters
curr_fraction = 0
curr_wealth = 1
curr_magnitude = curr_fraction * curr_wealth
curr_direction = np.zeros((dims, n_classes))

all_individual_cum_errors = []
for i, curr_point_idx in enumerate(range(n_points)):
    i = i + 1
    prev_direction = curr_direction
    prev_fraction = curr_fraction
    prev_wealth = curr_wealth
    prev_magnitude = curr_magnitude

    # update inner weight vector
    weight_vector = prev_magnitude * prev_direction

    # receive a new datapoint
    curr_x = features[curr_point_idx, :]
    curr_y = labels[curr_point_idx]

    all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector))

    # compute the gradient
    full_gradient = subgradient(curr_x, curr_y, weight_vector)

    # define inner step size
    inner_step_size = (1 / (L * R)) * np.sqrt(2 / i)

    # update inner direction
    curr_direction = l2_unit_ball_projection(prev_direction - inner_step_size * full_gradient)

    # update inner magnitude_wealth
    curr_wealth = prev_wealth - 1 / (R * L) * np.trace(full_gradient @ prev_direction.T) * prev_magnitude

    # update magnitude_betting_fraction
    curr_fraction = (1 / i) * ((i - 1) * prev_fraction - (1 / (L * R)) * np.trace(full_gradient @ prev_direction.T))

    # update magnitude
    curr_magnitude = curr_fraction * curr_wealth

    # plot stuff
    if i % 100 == 0:
        line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(),
                         c='tab:blue', label='parameter-free')
        plt.xlim(right=n_points)
        plt.xlabel('iterations', fontsize=24, fontweight="normal")
        plt.ylabel('cumulative error', fontsize=24, fontweight="normal")
        mypause(0.0001)

################################################################################################
################################################################################################
################################################################################################


# optimization for SGD
all_individual_cum_errors = []
step_size = 1
weights = np.random.randn(dims, n_classes)
for curr_point_idx in range(n_points):
    # receive a new datapoint
    curr_x = features[curr_point_idx, :]
    curr_y = labels[curr_point_idx]

    all_individual_cum_errors.append(loss(curr_x, curr_y, weights))

    # compute the gradient
    full_gradient = subgradient(curr_x, curr_y, weights)

    # update inner weight vector
    weights = weights - step_size * full_gradient

line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:red', label='SGD')
plt.xlim(right=n_points)
plt.xlabel('iterations', fontsize=24, fontweight="normal")
plt.ylabel('cumulative error', fontsize=24, fontweight="normal")
mypause(0.0001)

print(labels)
print(np.argmax(features @ weights, axis=1))

from sklearn.metrics import accuracy_score
print('')
print('accuracy: ', accuracy_score(labels, np.argmax(features @ weights, axis=1)))

plt.show()
k = 1