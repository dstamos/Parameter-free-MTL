import numpy as np
from numpy.linalg import norm
from src.general_functions import l2_unit_ball_projection, subgradient, loss
import matplotlib.pyplot as plt
import datetime
from src.general_functions import mypause
import time
import pandas as pd

# parameters
n_points = 8000
dims = 40
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

# range_wealth = np.linspace(0.0001, 1000, 400)
range_wealth = [1]

# optimization for parameter-free variation
L = 1
R = 1

temp_time = datetime.datetime.now()

all_accu_errors = np.zeros(len(range_wealth))
all_accu_errors[:] = np.nan

timestamp = datetime.datetime.now()
tt = time.time()
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
        curr_x = features[curr_point_idx, :]
        curr_y = labels[curr_point_idx]

        all_individual_cum_errors.append(loss(curr_x, curr_y, weight_vector, loss_name='absolute'))

        # compute the gradient
        subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
        full_gradient = subgrad * curr_x

        # define inner step size
        inner_step_size = (1 / (L * R)) * np.sqrt(2 / iteration)

        # update inner direction
        curr_direction = l2_unit_ball_projection(prev_direction - inner_step_size * full_gradient)

        # update inner magnitude_wealth
        curr_wealth = prev_wealth - 1 / (R * L) * full_gradient @ prev_direction * prev_magnitude

        # update magnitude_betting_fraction
        h_inner = (1 / (R * L)) * full_gradient @ prev_direction * (1 / (1 - (1 / (R * L)) * full_gradient @ prev_direction * prev_betting_fraction))
        all_h_inner.append(h_inner)
        a_thing_inner = 1 + np.sum([curr_h ** 2 for curr_h in all_h_inner])
        # update magnitude_betting_fraction
        # curr_fraction = np.max([np.min([prev_betting_fraction - (2 / (2 - np.log(3))) * (h_inner / a_thing_inner), 1 / 2]), -1 / 2])
        curr_fraction = (1 / iteration) * ((iteration - 1) * prev_betting_fraction - (1 / (L * R)) * (full_gradient @ prev_direction))

        # update magnitude
        curr_magnitude = curr_fraction * curr_wealth

        if curr_point_idx % 200 == 0:
            plt.plot(pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel(), c='tab:blue')
            plt.xlim(right=n_points)
            mypause(0.01)

    # print('aggressive progress: %6d/%6d %10.5f sec' % (initial_wealth_idx, len(range_wealth), time.time() - tt))
    # all_accu_errors[initial_wealth_idx] = np.nanmean(all_individual_cum_errors)
    # if initial_wealth_idx % 20 == 0:
    #     plt.clf()
    #     plt.plot(range_wealth, all_accu_errors, c='tab:blue')
    #     plt.xlim(right=max(range_wealth))
    #     plt.xlabel('initial wealth')
    #     plt.ylabel('final cumulative error')
    #     plt.title('cumulative error vs initial wealth')
    #     plt.tight_layout()
    #     plt.savefig('single_task_cumulative_error' + '_' + str(timestamp).replace(':', '') + '.png', format='png')
    #     mypause(0.01)

################################################################################################
################################################################################################
################################################################################################
