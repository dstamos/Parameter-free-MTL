import numpy as np
import pandas as pd
from src.general_functions import subgradient, loss
import matplotlib.pyplot as plt
from copy import deepcopy


class ParameterFreeAggressiveVariationOnlineStepSearch:
    def __init__(self):
        self.inner_step_size_range = [10 ** i for i in np.linspace(-3, 4, 10)]
        self.meta_step_size_range = [10 ** i for i in np.linspace(-3, 4, 10)]

    def fit(self, data):

        all_individual_cum_errors = []

        best_metaparameter = np.zeros(data.features_tr[0].shape[1])
        for task_iteration, task in enumerate(data.tr_task_indexes):
            x = data.features_tr[task]
            y = data.labels_tr[task]

            # initialize the inner parameters
            n_points, n_dims = x.shape

            curr_metaparameter = best_metaparameter

            best_perf = np.Inf
            for meta_idx, meta_step_size in enumerate(self.meta_step_size_range):
                for inner_idx, inner_step_size in enumerate(self.inner_step_size_range):

                    if meta_idx == 0 and inner_idx == 0:
                        og_metaparameter = deepcopy(curr_metaparameter)
                    else:
                        curr_metaparameter = og_metaparameter

                    temp_cum_errors = []
                    curr_untranslated_weights = np.zeros(n_dims)
                    shuffled_indexes = list(range(n_points))
                    # np.random.shuffle(shuffled_indexes)
                    for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                        prev_untranslated_weights = curr_untranslated_weights
                        prev_metaparameter = curr_metaparameter

                        # update inner weight vector
                        curr_weights = curr_untranslated_weights + prev_metaparameter

                        # receive a new datapoint
                        curr_x = x[curr_point_idx, :]
                        curr_y = y[curr_point_idx]

                        temp_cum_errors.append(loss(curr_x, curr_y, curr_weights, loss_name='absolute'))

                        # compute the gradient
                        subgrad = subgradient(curr_x, curr_y, curr_weights, loss_name='absolute')
                        full_gradient = subgrad * curr_x

                        # update metaparameters
                        curr_metaparameter = prev_metaparameter - meta_step_size * full_gradient

                        # update the untranslated weights
                        curr_untranslated_weights = prev_untranslated_weights - inner_step_size * full_gradient

                    curr_cum_sum = pd.DataFrame(temp_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
                    if curr_cum_sum[-1] < best_perf:
                        best_perf = curr_cum_sum[-1]
                        best_cum_errors = temp_cum_errors
                        best_metaparameter = deepcopy(curr_metaparameter)
                        print('%3d | best params: %10.5f | %10.5f | error: %5.3f' % (task_iteration, inner_step_size, meta_step_size, best_perf))
                        # print('inner step: %8e | meta step: %8e |       perf: %10.3f' % (inner_step_size, meta_step_size, np.nanmean(all_individual_cum_errors)))
                    else:
                        pass
                        # print('inner step: %8e | meta step: %8e | perf: %10.3f' % (inner_step_size, meta_step_size, np.nanmean(all_individual_cum_errors)))
            all_individual_cum_errors = all_individual_cum_errors + best_cum_errors
        return None, pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()

    @staticmethod
    def __general_iteration(t, n, i):
        return (t - 1) * n + i


class ParameterFreeLazyVariationOnlineStepSearch:
    def __init__(self):
        self.inner_step_size_range = [10 ** i for i in np.linspace(-8, 4, 20)]
        self.meta_step_size_range = [10 ** i for i in np.linspace(-8, 4, 20)]

    def fit(self, data):

        all_individual_cum_errors = []

        best_metaparameter = np.zeros(data.features_tr[0].shape[1])
        for task_iteration, task in enumerate(data.tr_task_indexes):

            x = data.features_tr[task]
            y = data.labels_tr[task]
            # initialize the inner parameters
            n_points, n_dims = x.shape

            prev_metaparameter = best_metaparameter

            best_perf = np.Inf
            for meta_idx, meta_step_size in enumerate(self.meta_step_size_range):
                for inner_idx, inner_step_size in enumerate(self.inner_step_size_range):

                    all_gradients = []

                    curr_untranslated_weights = np.zeros(n_dims)

                    temp_cum_errors = []
                    shuffled_indexes = list(range(n_points))
                    # np.random.shuffle(shuffled_indexes)
                    for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):

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
                    # print(len(all_individual_cum_errors))
                    # print('best params: %10e | %10e | error: %5.3f' % (inner_step_size, meta_step_size, curr_cum_sum[-1]))

                    if curr_cum_sum[-1] < best_perf:
                        best_perf = curr_cum_sum[-1]
                        best_cum_errors = temp_cum_errors
                        best_metaparameter = deepcopy(temp_metaparameter)
                        print('%3d | best params: %10e | %10e | error: %5.3f' % (task_iteration, inner_step_size, meta_step_size, best_perf))
                        # print('inner step: %8e | meta step: %8e |       perf: %10.3f' % (inner_step_size, meta_step_size, np.nanmean(all_individual_cum_errors)))
                    else:
                        pass
                        # print('inner step: %8e | meta step: %8e | perf: %10.3f' % (inner_step_size, meta_step_size, np.nanmean(all_individual_cum_errors)))
            all_individual_cum_errors = all_individual_cum_errors + best_cum_errors

        return None, pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
