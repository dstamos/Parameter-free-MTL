import numpy as np
from src.general_functions import l2_unit_ball_projection, subgradient, loss
import matplotlib.pyplot as plt


class ParameterFreeAggressive:
    def __init__(self, meta_initial_wealth, inner_initial_wealth, lipschitz_constant, input_norm_bound):
        self.lipschitz_constant = lipschitz_constant
        self.input_norm_bound = input_norm_bound
        self.all_weight_vectors = None
        self.all_meta_parameters = None
        self.meta_magnitude_betting_fraction = 0
        self.meta_magnitude_wealth = meta_initial_wealth
        self.inner_magnitude_betting_fraction = 0
        self.inner_magnitude_wealth = inner_initial_wealth

    def fit(self, x_list, y_list):
        n_dims = x_list[0].shape[1]
        curr_meta_magnitude_betting_fraction = self.meta_magnitude_betting_fraction
        curr_meta_magnitude_wealth = self.meta_magnitude_wealth
        curr_meta_magnitude = curr_meta_magnitude_betting_fraction * curr_meta_magnitude_wealth
        curr_meta_direction = np.zeros(n_dims)

        all_meta_parameters = []
        for task_iteration, (x, y) in enumerate(zip(x_list, y_list)):
            task_iteration = task_iteration + 1
            prev_meta_direction = curr_meta_direction
            prev_meta_magnitude_betting_fraction = curr_meta_magnitude_betting_fraction
            prev_meta_magnitude_wealth = curr_meta_magnitude_wealth
            prev_meta_magnitude = curr_meta_magnitude

            # initialize the inner parameters
            n_points, n_dims = x.shape
            curr_inner_magnitude_betting_fraction = self.inner_magnitude_betting_fraction
            curr_inner_magnitude_wealth = self.inner_magnitude_wealth
            curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth
            curr_inner_direction = np.zeros(n_dims)

            temp_weight_vectors = []
            shuffled_indexes = list(range(n_points))
            np.random.shuffle(shuffled_indexes)
            for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                inner_iteration = inner_iteration + 1
                prev_inner_direction = curr_inner_direction
                prev_inner_magnitude_betting_fraction = curr_inner_magnitude_betting_fraction
                prev_inner_magnitude_wealth = curr_inner_magnitude_wealth
                prev_inner_magnitude = curr_inner_magnitude

                # define total iteration
                total_iter = self.__general_iteration(task_iteration, n_points, inner_iteration)

                # update meta-parameter
                meta_parameter = prev_meta_magnitude * prev_meta_direction
                all_meta_parameters.append(meta_parameter)

                # update inner weight vector
                weight_vector = prev_inner_magnitude * prev_inner_direction + meta_parameter
                temp_weight_vectors.append(weight_vector)

                # receive a new datapoint
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]

                # compute the gradient
                subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='hinge')
                full_gradient = subgrad * curr_x

                # define meta step size
                meta_step_size = (1 / (self.lipschitz_constant * self.input_norm_bound)) * np.sqrt(2 / total_iter)

                # update meta-direction
                curr_meta_direction = l2_unit_ball_projection(prev_meta_direction - meta_step_size * full_gradient)

                # define inner step size
                inner_step_size = (1 / (self.lipschitz_constant * self.input_norm_bound)) * np.sqrt(2 / inner_iteration)

                # update inner direction
                curr_inner_direction = l2_unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

                # update meta-magnitude_wealth
                curr_meta_magnitude_wealth = prev_meta_magnitude_wealth - 1 / (self.input_norm_bound * self.lipschitz_constant) * full_gradient @ prev_meta_direction * prev_meta_magnitude

                # update meta-magnitude_betting_fraction
                curr_meta_magnitude_betting_fraction = - (1/total_iter) * ((total_iter - 1) * prev_meta_magnitude_betting_fraction + 1 / (self.lipschitz_constant * self.input_norm_bound) * full_gradient @ prev_meta_direction)

                # update meta-magnitude
                curr_meta_magnitude = curr_meta_magnitude_betting_fraction * curr_meta_magnitude_wealth

                # update inner magnitude_wealth
                curr_inner_magnitude_wealth = prev_inner_magnitude_wealth - 1 / (self.input_norm_bound * self.lipschitz_constant) * full_gradient @ prev_inner_direction * prev_inner_magnitude

                # update magnitude_betting_fraction
                curr_inner_magnitude_betting_fraction = -(1 / inner_iteration) * ((inner_iteration - 1) * prev_inner_magnitude_betting_fraction + 1 / (self.lipschitz_constant * self.input_norm_bound) * full_gradient @ prev_inner_direction)

                # update magnitude
                curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth

                print('%4d | magnitude: %6.3f' % (total_iter, curr_inner_magnitude))

            self.all_weight_vectors.append(np.mean(temp_weight_vectors, axis=0))
        self.all_meta_parameters = all_meta_parameters

    @staticmethod
    def __general_iteration(t, n, i):
        return (t - 1) * n + i

    @staticmethod
    def predict(x, w):
        # w = self.w
        y_pred = np.sign(x @ w)
        return y_pred
