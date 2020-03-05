import numpy as np
from numpy.linalg import norm
from src.general_functions import l2_unit_ball_projection, subgradient, loss
import matplotlib.pyplot as plt
import pandas as pd
from src.general_functions import mypause

# parameters
n_tasks = 10
n_points = 500
dims = 30
signal_to_noise_ratio = 10

seed = 9999
# FIXME Enable the line below for fixed results
np.random.seed(seed)

# data generation
all_features = [None] * n_tasks
all_labels = [None] * n_tasks

oracle = 10 * np.ones(dims)
for task_idx in range(n_tasks):
    # generating and normalizing the inputs
    features = np.random.randn(n_points, dims)
    features = features / norm(features, axis=1, keepdims=True)

    # generating and normalizing the weight vectors
    weight_vector = oracle  # + np.random.normal(loc=np.zeros(dims), scale=1).ravel()

    # generating labels and adding noise
    clean_labels = features @ weight_vector

    standard_noise = np.random.randn(n_points)
    noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
    labels = clean_labels + noise_std * standard_noise

    all_features[task_idx] = features
    all_labels[task_idx] = labels

my_dpi = 100
plt.figure(figsize=(1280 / my_dpi, 720 / my_dpi), facecolor='white', dpi=my_dpi)
plt.title('blue: parameter-free | red: SGD')
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')
plt.pause(0.001)

################################################################################################
################################################################################################
################################################################################################


# optimization

L = 1
R = 1

curr_meta_fraction = 0
curr_meta_wealth = 1
curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth
curr_meta_direction = np.zeros(dims)

all_h_meta = []
total_iter = 0
all_individual_cum_errors = []
for task_iteration, task in enumerate(range(n_tasks)):
    task_iteration = task_iteration + 1

    x = all_features[task]
    y = all_labels[task]

    prev_meta_direction = curr_meta_direction
    prev_meta_fraction = curr_meta_fraction
    prev_meta_wealth = curr_meta_wealth
    prev_meta_magnitude = curr_meta_magnitude

    # initialize the inner parameters
    curr_inner_fraction = 0
    curr_inner_wealth = 1
    curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth
    curr_inner_direction = np.zeros(dims)

    all_h_inner = []
    for inner_iteration, curr_point_idx in enumerate(range(n_points)):
        inner_iteration = inner_iteration + 1

        prev_inner_direction = curr_inner_direction
        prev_inner_fraction = curr_inner_fraction
        prev_inner_wealth = curr_inner_wealth
        prev_inner_magnitude = curr_inner_magnitude

        # define total iteration
        total_iter = total_iter + 1

        # update meta-parameter
        meta_parameter = prev_meta_magnitude * prev_meta_direction

        # update inner weight vector
        # FIXME Split me?
        weight_vector = prev_inner_magnitude * prev_inner_direction + meta_parameter

        # receive a new datapoint
        curr_x = x[curr_point_idx, :]
        curr_y = y[curr_point_idx]

        all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector, loss_name='absolute'))

        # compute the gradient
        subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
        full_gradient = subgrad * curr_x

        # define meta step size
        meta_step_size = (1 / (L * R)) * np.sqrt(2 / total_iter)

        # update meta-direction
        curr_meta_direction = l2_unit_ball_projection(prev_meta_direction - meta_step_size * full_gradient)

        # define inner step size
        inner_step_size = (1 / (L * R)) * np.sqrt(2 / inner_iteration)

        # update inner direction
        curr_inner_direction = l2_unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

        # update meta-magnitude_wealth
        curr_meta_wealth = prev_meta_wealth - (prev_meta_magnitude / (R * L)) * (full_gradient @ prev_meta_direction)

        # update meta-magnitude_betting_fraction
        # curr_meta_fraction = (1/total_iter) * ((total_iter-1) * prev_meta_fraction - (1/(L * R))*(full_gradient @ prev_meta_direction))

        h_inner = (1 / (R * L)) * full_gradient @ prev_meta_direction * (1 / (1 - (1 / (R * L)) * full_gradient @ prev_meta_direction * prev_meta_fraction))
        all_h_meta.append(h_inner)
        a_thing_meta = 1 + np.sum([curr_h ** 2 for curr_h in all_h_meta])
        curr_meta_fraction = np.max([np.min([prev_meta_fraction - (2 / (2 - np.log(3))) * (h_inner / a_thing_meta), 1 / 2]), -1 / 2])

        # update meta-magnitude
        curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth

        # update inner magnitude_wealth
        curr_inner_wealth = prev_inner_wealth - (prev_inner_magnitude / (R * L)) * (full_gradient @ prev_inner_direction)

        # update magnitude_betting_fraction
        # curr_inner_fraction = (1 / inner_iteration) * ((inner_iteration - 1) * prev_inner_fraction - (1 / (L * R)) * (full_gradient @ prev_inner_direction))

        h_inner = (1 / (R * L)) * full_gradient @ prev_inner_direction * (1 / (1 - (1 / (R * L)) * full_gradient @ prev_inner_direction * prev_inner_fraction))
        all_h_inner.append(h_inner)
        a_thing_inner = 1 + np.sum([curr_h ** 2 for curr_h in all_h_inner])
        curr_inner_fraction = np.max([np.min([prev_inner_fraction - (2 / (2 - np.log(3))) * (h_inner / a_thing_inner), 1 / 2]), -1 / 2])

        # update magnitude
        curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth

        # plot stuff
        if total_iter % 500 == 0:
            line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:blue', label='parameter-free')
            plt.xlim(right=n_points*n_tasks)
            plt.xlabel('iterations', fontsize=38, fontweight="normal")
            plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
            mypause(0.0001)

line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:blue', label='parameter-free')
plt.xlim(right=n_points*n_tasks)
plt.xlabel('iterations', fontsize=38, fontweight="normal")
plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
mypause(0.0001)
################################################################################################
################################################################################################
################################################################################################

# optimization for SGD (concatenated tasks)
concat_features = np.concatenate(all_features, axis=0)
concat_labels = np.concatenate(all_labels, axis=0)

n_points, dims = concat_features.shape

range_wealth = [1]

# optimization for parameter-free variation
L = 1
R = 1

for initial_wealth_idx, initial_wealth in enumerate(range_wealth):
    all_individual_cum_errors = []

    # initialize the inner parameters
    curr_betting_fraction = 0
    curr_wealth = initial_wealth
    curr_magnitude = curr_betting_fraction * curr_wealth
    curr_direction = np.zeros(dims)

    all_h_inner = []

    for iteration, curr_point_idx in enumerate(range(n_points)):
        iteration = iteration + 1
        prev_direction = curr_direction
        prev_betting_fraction = curr_betting_fraction
        prev_wealth = curr_wealth
        prev_magnitude = curr_magnitude

        # update inner weight vector
        weight_vector = prev_magnitude * prev_direction

        # receive a new datapoint
        curr_x = concat_features[curr_point_idx, :]
        curr_y = concat_labels[curr_point_idx]

        all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector, loss_name='absolute'))

        # compute the gradient
        subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
        full_gradient = subgrad * curr_x

        # define inner step size
        inner_step_size = (1 / (L * R)) * np.sqrt(2 / iteration)

        # update inner direction
        curr_direction = l2_unit_ball_projection(prev_direction - inner_step_size * full_gradient)

        # update inner magnitude_wealth
        curr_wealth = prev_wealth - (prev_magnitude / (R * L)) * (full_gradient @ prev_direction)

        # update magnitude_betting_fraction
        h_inner = (1 / (R * L)) * full_gradient @ prev_direction * (1 / (1 - (1 / (R * L)) * full_gradient @ prev_direction * prev_betting_fraction))
        all_h_inner.append(h_inner)
        a_thing_inner = 1 + np.sum([curr_h ** 2 for curr_h in all_h_inner])
        # update magnitude_betting_fraction
        curr_betting_fraction = np.max([np.min([prev_betting_fraction - (2 / (2 - np.log(3))) * (h_inner / a_thing_inner), 1 / 2]), -1 / 2])
        # curr_betting_fraction = (1 / iteration) * ((iteration - 1) * prev_betting_fraction - (1 / (L * R)) * (full_gradient @ prev_direction))

        # update magnitude
        curr_magnitude = curr_betting_fraction * curr_wealth

        if curr_point_idx % 1000 == 0:
            plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:red')
            plt.xlim(right=n_points)
            mypause(0.01)
plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:red')
# plt.xlim(right=500)
mypause(0.01)

plt.show()