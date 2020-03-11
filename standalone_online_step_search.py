import numpy as np
from numpy.linalg import norm
from src.general_functions import l2_unit_ball_projection, subgradient, loss
import matplotlib.pyplot as plt
import pandas as pd
from src.general_functions import mypause
from copy import deepcopy
import time

# parameters
n_tasks = 100
n_points = 20
dims = 30
signal_to_noise_ratio = 10

seed = 9999
np.random.seed(seed)

inner_step_size_range = [10 ** i for i in np.linspace(-10, 4, 24)]
meta_step_size_range = [10 ** i for i in np.linspace(-10, 4, 24)]

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
plt.figure(figsize=(1080 / my_dpi, 720 / my_dpi), facecolor='white', dpi=my_dpi)
mypause(0.0001)
################################################################################################
################################################################################################
################################################################################################

# Aggressive step search

# best_perf = np.Inf
# for inner_idx, inner_step_size in enumerate(inner_step_size_range):
#     for meta_idx, meta_step_size in enumerate(meta_step_size_range):
#         all_individual_cum_errors = []
#
#         curr_metaparameter = np.zeros(dims)
#         for task_iteration, task in enumerate(range(n_tasks)):
#             x = all_features[task]
#             y = all_labels[task]
#
#             task_iteration = task_iteration + 1
#
#             # initialize the inner parameters
#             n_points, n_dims = x.shape
#
#             curr_untranslated_weights = np.zeros(n_dims)
#             shuffled_indexes = list(range(n_points))
#             # np.random.shuffle(shuffled_indexes)
#             for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
#                 prev_untranslated_weights = curr_untranslated_weights
#                 prev_metaparameter = curr_metaparameter
#
#                 # update inner weight vector
#                 curr_weights = curr_untranslated_weights + curr_metaparameter
#
#                 # receive a new datapoint
#                 curr_x = x[curr_point_idx, :]
#                 curr_y = y[curr_point_idx]
#
#                 all_individual_cum_errors.append(loss(curr_x, curr_y, curr_weights, loss_name='absolute'))
#
#                 # compute the gradient
#                 subgrad = subgradient(curr_x, curr_y, curr_weights, loss_name='absolute')
#                 full_gradient = subgrad * curr_x
#
#                 # update metaparameters
#                 curr_metaparameter = prev_metaparameter - meta_step_size * full_gradient
#
#                 # update the untranslated weights
#                 curr_untranslated_weights = prev_untranslated_weights - inner_step_size * full_gradient
#
#         average_stuff = pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
#         if average_stuff[-1] < best_perf:
#             best_perf = average_stuff[-1]
#             best_average = average_stuff
#
# line, = plt.plot(best_average, linestyle='--', c='tab:blue', label='aggressive step search')
# plt.xlim(right=n_points*n_tasks)
# plt.xlabel('iterations', fontsize=38, fontweight="normal")
# plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
# mypause(0.0001)

################################################################################################
################################################################################################
################################################################################################

# Aggressive online step search

# tt = time.time()
#
# all_individual_cum_errors = []
#
# best_metaparameter = np.zeros(dims)
# for task_iteration, task in enumerate(range(n_tasks)):
#     x = all_features[task]
#     y = all_labels[task]
#
#     # initialize the inner parameters
#     n_points, n_dims = x.shape
#
#     curr_metaparameter = best_metaparameter
#
#     best_perf = np.Inf
#     for meta_idx, meta_step_size in enumerate(meta_step_size_range):
#         for inner_idx, inner_step_size in enumerate(inner_step_size_range):
#
#             if meta_idx == 0 and inner_idx == 0:
#                 og_metaparameter = deepcopy(curr_metaparameter)
#             else:
#                 curr_metaparameter = og_metaparameter
#
#             temp_cum_errors = []
#             curr_untranslated_weights = np.zeros(n_dims)
#             shuffled_indexes = list(range(n_points))
#             # np.random.shuffle(shuffled_indexes)
#             for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
#                 prev_untranslated_weights = curr_untranslated_weights
#                 prev_metaparameter = curr_metaparameter
#
#                 # update inner weight vector
#                 curr_weights = curr_untranslated_weights + prev_metaparameter
#
#                 # receive a new datapoint
#                 curr_x = x[curr_point_idx, :]
#                 curr_y = y[curr_point_idx]
#
#                 temp_cum_errors.append(loss(curr_x, curr_y, curr_weights, loss_name='absolute'))
#
#                 # compute the gradient
#                 subgrad = subgradient(curr_x, curr_y, curr_weights, loss_name='absolute')
#                 full_gradient = subgrad * curr_x
#
#                 # update metaparameters
#                 curr_metaparameter = prev_metaparameter - meta_step_size * full_gradient
#
#                 # update the untranslated weights
#                 curr_untranslated_weights = prev_untranslated_weights - inner_step_size * full_gradient
#
#             curr_cum_sum = pd.DataFrame(temp_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
#             if curr_cum_sum[-1] < best_perf:
#                 best_perf = curr_cum_sum[-1]
#                 best_cum_errors = temp_cum_errors
#                 best_metaparameter = deepcopy(curr_metaparameter)
#     all_individual_cum_errors = all_individual_cum_errors + best_cum_errors
#     print('task: %4d | %5.2f sec' % (task_iteration, time.time() - tt))
# line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), linewidth=2, linestyle=':', c='blue', label='aggressive online step search')
# mypause(0.01)

################################################################################################
################################################################################################
################################################################################################

# Lazy step search

best_perf = np.Inf
for _, inner_step_size in enumerate(inner_step_size_range):
    for _, meta_step_size in enumerate(meta_step_size_range):
        all_individual_cum_errors = []
        best_mtl_performances = []

        curr_metaparameter = np.zeros(dims)
        for task_iteration, task in enumerate(range(n_tasks)):
            x = all_features[task]
            y = all_labels[task]

            prev_metaparameter = curr_metaparameter
            all_gradients = []
            # initialize the inner parameters
            n_points, n_dims = x.shape

            curr_untranslated_weights = np.zeros(n_dims)

            shuffled_indexes = list(range(n_points))
            # np.random.shuffle(shuffled_indexes)
            for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                prev_untranslated_weights = curr_untranslated_weights

                # update inner weight vector
                curr_weights = curr_untranslated_weights + curr_metaparameter

                # receive a new datapoint
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]

                all_individual_cum_errors.append(loss(curr_x, curr_y, curr_weights, loss_name='absolute'))

                # compute the gradient
                subgrad = subgradient(curr_x, curr_y, curr_weights, loss_name='absolute')
                full_gradient = subgrad * curr_x
                all_gradients.append(full_gradient)

                # update the untranslated weights
                curr_untranslated_weights = prev_untranslated_weights - inner_step_size * full_gradient

            # update metaparameters
            curr_metaparameter = prev_metaparameter - meta_step_size * np.sum(all_gradients, axis=0)

        average_stuff = pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
        if average_stuff[-1] < best_perf:
            best_perf = average_stuff[-1]
            best_inner = inner_step_size
            best_meta = meta_step_size
            best_average = average_stuff

line, = plt.plot(best_average, linestyle='--', c='tab:red', label='lazy step search')
plt.xlim(right=n_points*n_tasks)
plt.xlabel('iterations', fontsize=38, fontweight="normal")
plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
mypause(0.0001)

################################################################################################
################################################################################################
################################################################################################

# Lazy online step search

tt = time.time()
all_individual_cum_errors = []

best_metaparameter = np.zeros(dims)
for task_iteration, task in enumerate(range(n_tasks)):

    x = all_features[task]
    y = all_labels[task]
    # initialize the inner parameters
    n_points = x.shape[0]

    prev_metaparameter = best_metaparameter

    best_perf = np.Inf
    for meta_idx, meta_step_size in enumerate(meta_step_size_range):
        for inner_idx, inner_step_size in enumerate(inner_step_size_range):

            all_gradients = []

            curr_untranslated_weights = np.zeros(dims)

            temp_cum_errors = []
            # np.random.shuffle(shuffled_indexes)
            for inner_iteration, curr_point_idx in enumerate(range(n_points)):
                prev_untranslated_weights = curr_untranslated_weights

                # update inner weight vector
                curr_weights = curr_untranslated_weights + prev_metaparameter

                # receive a new datapoint
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]

                temp_cum_errors.append(loss(curr_x, curr_y, curr_weights, loss_name='absolute'))

                # compute the gradient
                subgrad = subgradient(curr_x, curr_y, curr_weights, loss_name='absolute')
                full_gradient = subgrad * curr_x
                all_gradients.append(full_gradient)

                # update the untranslated weights
                curr_untranslated_weights = prev_untranslated_weights - inner_step_size * full_gradient

            # update metaparameters
            temp_metaparameter = prev_metaparameter - meta_step_size * np.sum(all_gradients, axis=0)

            curr_cum_sum = pd.DataFrame(all_individual_cum_errors + temp_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()

            if curr_cum_sum[-1] < best_perf:
                best_perf = curr_cum_sum[-1]
                best_cum_errors = temp_cum_errors
                best_metaparameter = deepcopy(temp_metaparameter)
    all_individual_cum_errors = all_individual_cum_errors + best_cum_errors
    print('task: %4d | %5.2f sec' % (task_iteration, time.time() - tt))
line, = plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), linewidth=2, linestyle='-.', c='red', label='lazy online step search')

plt.legend()
plt.show()