import numpy as np
from numpy.linalg import norm
from src.general_functions import l2_unit_ball_projection, subgradient, loss
import matplotlib.pyplot as plt
import pandas as pd
from src.general_functions import mypause

# parameters
n_points = 2000
dims = 30
signal_to_noise_ratio = 10

seed = 9999
# FIXME Enable the line below for fixed results
np.random.seed(seed)

# data generation
oracle = 10 * np.ones(dims)
# generating and normalizing the inputs
features = np.random.randn(n_points, dims)
features = features / norm(features, axis=1, keepdims=True)

# generating and normalizing the weight vectors
weight_vector = oracle + np.random.normal(loc=np.zeros(dims), scale=1).ravel()

# generating labels and adding noise
clean_labels = features @ weight_vector

standard_noise = np.random.randn(n_points)
noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
labels = clean_labels + noise_std * standard_noise

my_dpi = 100
plt.figure(figsize=(1280 / my_dpi, 720 / my_dpi), facecolor='white', dpi=my_dpi)
plt.title('blue: parameter-free | red: SGD')
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.pause(0.001)

################################################################################################
################################################################################################
################################################################################################

# optimization for parameter free
L = 1
R = 1

# initialize the inner parameters
curr_fraction = 0
curr_wealth = 1
curr_magnitude = curr_fraction * curr_wealth
curr_direction = np.zeros(dims)

all_individual_cum_errors = []
for inner_iteration, curr_point_idx in enumerate(range(n_points)):
    inner_iteration = inner_iteration + 1
    print(inner_iteration)
    prev_direction = curr_direction
    prev_fraction = curr_fraction
    prev_wealth = curr_wealth
    prev_magnitude = curr_magnitude

    # update inner weight vector
    weight_vector = prev_magnitude * prev_direction

    # receive a new datapoint
    curr_x = features[curr_point_idx, :]
    curr_y = labels[curr_point_idx]

    all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector, loss_name='absolute'))

    # compute the gradient
    subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
    full_gradient = subgrad * curr_x

    # define inner step size
    inner_step_size = (1 / (L * R)) * np.sqrt(2 / inner_iteration)

    # update inner direction
    curr_direction = l2_unit_ball_projection(prev_direction - inner_step_size * full_gradient)

    # update inner magnitude_wealth
    curr_wealth = prev_wealth - 1 / (R * L) * full_gradient @ prev_direction * prev_magnitude

    # update magnitude_betting_fraction
    curr_fraction = -(1 / inner_iteration) * ((inner_iteration - 1) * prev_fraction + (1 / (L * R)) * (full_gradient @ prev_direction))

    # update magnitude
    curr_magnitude = curr_fraction * curr_wealth

    # plot stuff
    if inner_iteration % 100 == 0:
        line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:blue', label='parameter-free')
        plt.xlim(right=n_points)
        plt.xlabel('iterations', fontsize=38, fontweight="normal")
        plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
        mypause(0.0001)

################################################################################################
################################################################################################
################################################################################################

# optimization for SGD
all_individual_cum_errors = []
step_size = 1e-4
all_individual_cum_errors_gd = []
curr_weights = np.random.randn(dims)
for inner_iteration, curr_point_idx in enumerate(range(n_points)):
    inner_iteration = inner_iteration + 1
    prev_weights = curr_weights

    # receive a new datapoint
    curr_x = features[curr_point_idx, :]
    curr_y = labels[curr_point_idx]

    all_individual_cum_errors.append(loss(curr_x, curr_y, prev_weights, loss_name='absolute'))

    # compute the gradient
    subgrad = subgradient(curr_x, curr_y, prev_weights, loss_name='absolute')
    full_gradient = subgrad * curr_x

    # update inner weight vector
    weight_vector = prev_weights - step_size * full_gradient

    # plot stuff
    if inner_iteration % 100 == 0:
        line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:red', label='SGD')
        plt.xlim(right=n_points)
        plt.xlabel('iterations', fontsize=38, fontweight="normal")
        plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
        mypause(0.0001)
plt.show()
k = 1