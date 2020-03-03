import numpy as np
from numpy.linalg import norm
from src.general_functions import l2_unit_ball_projection, subgradient, loss
import matplotlib.pyplot as plt
import pandas as pd
from src.general_functions import mypause

# parameters
n_tasks = 100
n_points = 200
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
    weight_vector = oracle + np.random.normal(loc=np.zeros(dims), scale=1).ravel()

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
def general_iteration(t, n, i):
    return (t - 1) * n + i


L = 1
R = 1

curr_meta_fraction = 0
curr_meta_wealth = 1
curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth
curr_meta_direction = np.zeros(dims)

all_final_weight_vectors = []
all_individual_cum_errors = []

all_mtl_performances = []
all_meta_parameters = []
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

    temp_weight_vectors = []
    for inner_iteration, curr_point_idx in enumerate(range(n_points)):
        inner_iteration = inner_iteration + 1
        prev_inner_direction = curr_inner_direction
        prev_inner_fraction = curr_inner_fraction
        prev_inner_wealth = curr_inner_wealth
        prev_inner_magnitude = curr_inner_magnitude

        # define total iteration
        total_iter = general_iteration(task_iteration, n_points, inner_iteration)

        # update meta-parameter
        meta_parameter = prev_meta_magnitude * prev_meta_direction
        all_meta_parameters.append(meta_parameter)

        # update inner weight vector
        weight_vector = prev_inner_magnitude * prev_inner_direction + meta_parameter
        temp_weight_vectors.append(weight_vector)

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
        curr_meta_wealth = prev_meta_wealth - (1 / (R * L)) * full_gradient @ prev_meta_direction * prev_meta_magnitude

        # update meta-magnitude_betting_fraction
        curr_meta_fraction = -(1/total_iter) * ((total_iter-1) * prev_meta_fraction + (1/(L * R))*(full_gradient @ prev_meta_direction))

        # update meta-magnitude
        curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth

        # update inner magnitude_wealth
        curr_inner_wealth = prev_inner_wealth - 1 / (R * L) * full_gradient @ prev_inner_direction * prev_inner_magnitude

        # update magnitude_betting_fraction
        curr_inner_fraction = -(1 / inner_iteration) * ((inner_iteration - 1) * prev_inner_fraction + (1 / (L * R)) * (full_gradient @ prev_inner_direction))

        # update magnitude
        curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth

        # plot stuff
        if inner_iteration % 100 == 0:
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

all_individual_cum_errors = []
step_size = 1e-4
all_individual_cum_errors_gd = []
curr_weights = np.random.randn(dims)
for inner_iteration, curr_point_idx in enumerate(range(len(concat_labels))):
    inner_iteration = inner_iteration + 1
    prev_weights = curr_weights

    # receive a new datapoint
    curr_x = concat_features[curr_point_idx, :]
    curr_y = concat_labels[curr_point_idx]

    all_individual_cum_errors.append(loss(curr_x, curr_y, prev_weights, loss_name='absolute'))

    # compute the gradient
    subgrad = subgradient(curr_x, curr_y, prev_weights, loss_name='absolute')
    full_gradient = subgrad * curr_x

    # update inner weight vector
    weight_vector = prev_weights - step_size * full_gradient

    # plot stuff
    if inner_iteration % 100 == 0:
        line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:red', label='SGD')
        plt.xlabel('iterations', fontsize=38, fontweight="normal")
        plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
        mypause(0.0001)
plt.show()
k = 1


k = 1