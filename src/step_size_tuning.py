import numpy as np
import pandas as pd
from src.general_functions import subgradient, loss
import matplotlib.pyplot as plt


class ParameterFreeAggressiveVariationStepSearch:
    def __init__(self):
        self.inner_step_size_range = [10 ** i for i in np.linspace(-3, 4, 12)]
        self.meta_step_size_range = [10 ** i for i in np.linspace(-3, 4, 12)]

    def fit(self, data):

        best_perf = np.Inf
        for _, inner_step_size in enumerate(self.inner_step_size_range):
            for _, meta_step_size in enumerate(self.meta_step_size_range):
                all_individual_cum_errors = []
                all_mtl_performances = []
                all_final_weight_vectors = []

                curr_metaparameter = np.zeros(data.features_tr[0].shape[1])
                for task_iteration, task in enumerate(data.tr_task_indexes):
                    x = data.features_tr[task]
                    y = data.labels_tr[task]

                    task_iteration = task_iteration + 1

                    # initialize the inner parameters
                    n_points, n_dims = x.shape

                    curr_untranslated_weights = np.zeros(n_dims)
                    temp_weight_vectors = []
                    shuffled_indexes = list(range(n_points))
                    # np.random.shuffle(shuffled_indexes)
                    for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                        prev_untranslated_weights = curr_untranslated_weights
                        prev_metaparameter = curr_metaparameter

                        # update inner weight vector
                        curr_weights = curr_untranslated_weights + curr_metaparameter
                        temp_weight_vectors.append(curr_weights)

                        # receive a new datapoint
                        curr_x = x[curr_point_idx, :]
                        curr_y = y[curr_point_idx]

                        all_individual_cum_errors.append(loss(curr_x, curr_y, curr_weights, loss_name='absolute'))

                        # compute the gradient
                        subgrad = subgradient(curr_x, curr_y, curr_weights, loss_name='absolute')
                        full_gradient = subgrad * curr_x

                        # update metaparameters
                        curr_metaparameter = prev_metaparameter - meta_step_size * full_gradient

                        # update the untranslated weights
                        curr_untranslated_weights = prev_untranslated_weights - inner_step_size * full_gradient

                    all_final_weight_vectors.append(np.mean(temp_weight_vectors, axis=0))
                    all_test_errors = []
                    for idx, curr_test_task in enumerate(data.tr_task_indexes[:task_iteration]):
                        all_test_errors.append(loss(data.features_ts[curr_test_task], data.labels_ts[curr_test_task], all_final_weight_vectors[idx], loss_name='absolute'))
                    all_mtl_performances.append(np.nanmean(all_test_errors))

                average_stuff = pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
                if average_stuff[-1] < best_perf:
                    best_perf = average_stuff[-1]
                    best_mtl_performances = all_mtl_performances
                    best_average = average_stuff
                    print('inner step: %8e | meta step: %8e |       perf: %10.3f' % (inner_step_size, meta_step_size, np.nanmean(all_individual_cum_errors)))
                else:
                    print('inner step: %8e | meta step: %8e | perf: %10.3f' % (inner_step_size, meta_step_size, np.nanmean(all_individual_cum_errors)))
        return best_mtl_performances, best_average

    @staticmethod
    def __general_iteration(t, n, i):
        return (t - 1) * n + i


class ParameterFreeLazyVariationStepSearch:
    def __init__(self):
        self.inner_step_size_range = [10 ** i for i in np.linspace(-3, 1, 6)]
        self.meta_step_size_range = [10 ** i for i in np.linspace(-3, 1, 6)]

    def fit(self, data):

        best_perf = np.Inf
        for _, inner_step_size in enumerate(self.inner_step_size_range):
            for _, meta_step_size in enumerate(self.meta_step_size_range):
                all_individual_cum_errors = []
                best_mtl_performances = []

                curr_metaparameter = np.zeros(data.features_tr[0].shape[1])
                for task_iteration, task in enumerate(data.tr_task_indexes):
                    x = data.features_tr[task]
                    y = data.labels_tr[task]

                    prev_metaparameter = curr_metaparameter
                    temp_weight_vectors = []
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
                        temp_weight_vectors.append(curr_weights)

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

                    curr_test_perf = loss(data.features_ts[task], data.labels_ts[task], np.mean(temp_weight_vectors, axis=0), loss_name='absolute')
                    best_mtl_performances.append(curr_test_perf)

                    # update metaparameters
                    curr_metaparameter = prev_metaparameter - meta_step_size * np.sum(all_gradients, axis=0)

                average_stuff = pd.DataFrame(all_individual_cum_errors).rolling(window=10 ** 10, min_periods=1).mean().values.ravel()
                if average_stuff[-1] < best_perf:
                    best_perf = average_stuff[-1]
                    best_inner = inner_step_size
                    best_meta = meta_step_size
                    best_average = average_stuff
                # plt.plot(average_stuff)
                # plt.title('best inner step ' + str(best_inner) + ' | ' + 'best meta step ' + str(best_meta))
                # plt.ylim(top=12, bottom=0)
                # plt.pause(0.1)
                # k = 1

        return None, best_average
