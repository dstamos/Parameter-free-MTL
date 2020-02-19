import numpy as np
from src.general_functions import l2_unit_ball_projection, subgradient, loss
import matplotlib.pyplot as plt


class BasicBias:
    def __init__(self, fixed_bias):
        self.fixed_bias = fixed_bias
        self.step_size_range = [10**i for i in np.linspace(-1, 5, 200)]
        self.w = None

    def fit(self, x, y):
        n_points, n_dims = x.shape

        for step_idx, step_size in enumerate(self.step_size_range):
            curr_untranslated_weights = np.zeros(n_dims)
            curr_weights = curr_untranslated_weights + self.fixed_bias
            all_weight_vectors = []
            all_losses = []
            shuffled_indexes = list(range(n_points))
            np.random.shuffle(shuffled_indexes)
            for iteration, curr_point_idx in enumerate(shuffled_indexes):
                prev_untranslated_weights = curr_untranslated_weights
                prev_weights = curr_weights

                # receive a new datapoint
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]

                # compute the gradient
                subgrad = subgradient(curr_x, curr_y, prev_weights, loss_name='absolute')
                full_gradient = subgrad * curr_x

                # update weight vector
                curr_untranslated_weights = prev_untranslated_weights - step_size * full_gradient
                curr_weights = curr_untranslated_weights + self.fixed_bias
                all_weight_vectors.append(curr_weights)

                if len(all_weight_vectors) < 2:
                    final_w = curr_weights
                else:
                    final_w = np.mean(all_weight_vectors, axis=0)
                loss_thing = loss(x, y, final_w, loss_name='absolute')
                all_losses.append(loss_thing)

            # update the final vector w
            plt.plot(all_losses)
            plt.ylim(bottom=0, top=1)
            plt.pause(0.1)
            self.w = np.mean(all_weight_vectors, axis=0)
        plt.show()

    def predict(self, x):
        w = self.w
        y_pred = np.sign(x @ w)
        return y_pred


class ParameterFreeFixedBias:
    def __init__(self, fixed_bias, initial_wealth, lipschitz_constant, input_norm_bound):
        self.L = lipschitz_constant
        self.R = input_norm_bound
        self.fixed_bias = fixed_bias
        self.w = None
        self.magnitude_betting_fraction = 0
        self.magnitude_wealth = initial_wealth

    def fit(self, x, y):
        n_points, n_dims = x.shape

        range_shit = [10**i for i in np.linspace(-5, 5, 200)]
        # range_shit = np.linspace(0.1, n_points, 50)
        # range_shit = [1]
        for idx, value_shit in enumerate(range_shit):
            curr_bet_fraction = self.magnitude_betting_fraction
            curr_wealth = value_shit   # self.magnitude_wealth
            curr_magnitude = curr_bet_fraction * curr_wealth
            curr_direction = np.zeros(n_dims)

            all_losses = []
            all_weight_vectors = []

            shuffled_indexes = list(range(n_points))
            np.random.shuffle(shuffled_indexes)
            for iter, curr_point_idx in enumerate(shuffled_indexes):
                iter = iter + 1
                prev_direction = curr_direction
                prev_bet_fraction = curr_bet_fraction
                prev_wealth = curr_wealth
                prev_magnitude = curr_magnitude

                # update weight vector
                weight_vector = prev_magnitude * prev_direction + self.fixed_bias
                all_weight_vectors.append(weight_vector)

                # receive a new datapoint
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]

                # compute the gradient
                subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
                full_gradient = subgrad * curr_x

                # define step size
                step_size = 1 / (self.L * self.R) * np.sqrt(2 / iter)

                # update direction
                curr_direction = l2_unit_ball_projection(prev_direction - step_size * full_gradient)

                # update magnitude_wealth
                curr_wealth = prev_wealth - (1 / (self.R * self.L)) * full_gradient @ prev_direction * prev_magnitude

                # update magnitude_betting_fraction
                curr_bet_fraction = -(1 / iter) * ((iter - 1) * prev_bet_fraction + (1 / (self.L * self.R)) * full_gradient @ prev_direction)

                # update magnitude
                curr_magnitude = curr_bet_fraction * curr_wealth

                if len(all_weight_vectors) < 2:
                    final_w = weight_vector
                else:
                    final_w = np.mean(all_weight_vectors, axis=0)
                loss_thing = loss(x, y, final_w, loss_name='absolute')
                all_losses.append(loss_thing)
            print('initial wealth %5f | error %5.2f' % (value_shit, loss_thing))
            print('')
            plt.plot(all_losses)
            plt.ylim(bottom=0, top=5)
            plt.pause(0.1)
            # update the final vector w
            self.w = np.mean(all_weight_vectors, axis=0)
            print(self.w)
        plt.show()

    def predict(self, x):
        w = self.w
        y_pred = np.sign(x @ w)
        return y_pred


class ParameterFreeFixedBiasVariation:
    def __init__(self, fixed_bias, initial_wealth, lipschitz_constant, input_norm_bound):
        self.L = lipschitz_constant
        self.R = input_norm_bound
        self.fixed_bias = fixed_bias
        self.w = None
        self.magnitude_betting_fraction = 0
        self.magnitude_wealth = initial_wealth

    def fit(self, x, y):
        n_points, n_dims = x.shape

        # range_shit = [10**i for i in np.linspace(-6, 6, 100)]
        # range_shit = np.linspace(0.1, n_points, 20)
        range_shit = [np.sqrt(n_points)]
        for idx, value_shit in enumerate(range_shit):
            curr_bet_fraction = self.magnitude_betting_fraction

            from copy import deepcopy
            initial_shit = deepcopy(value_shit)

            curr_wealth = value_shit   # self.magnitude_wealth
            curr_magnitude = curr_bet_fraction * curr_wealth
            curr_direction = np.zeros(n_dims)

            all_losses = []
            all_weight_vectors = []

            all_h = []

            shuffled_indexes = list(range(n_points))
            np.random.shuffle(shuffled_indexes)
            for iteration, curr_point_idx in enumerate(shuffled_indexes):
                iteration = iteration + 1
                prev_direction = curr_direction
                prev_bet_fraction = curr_bet_fraction
                prev_wealth = curr_wealth
                prev_magnitude = curr_magnitude

                # update weight vector
                weight_vector = prev_magnitude * prev_direction + self.fixed_bias
                all_weight_vectors.append(weight_vector)

                # receive a new datapoint
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]

                # compute the gradient
                subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
                full_gradient = subgrad * curr_x

                # define step size
                step_size = 1 / (self.L * self.R) * np.sqrt(2 / iteration)

                # update direction
                curr_direction = l2_unit_ball_projection(prev_direction - step_size * full_gradient)

                # update magnitude_wealth
                curr_wealth = prev_wealth - (1 / (self.R * self.L)) * full_gradient @ prev_direction * prev_magnitude

                # h_thing
                h = (1 / (self.R * self.L)) * full_gradient @ prev_direction * (1 / (1 - (1 / (self.R * self.L)) * full_gradient @ prev_direction * prev_bet_fraction))
                all_h.append(h)
                A_thing = 1 + np.sum([curr_h**2 for curr_h in all_h])

                # update magnitude_betting_fraction
                curr_bet_fraction = np.max([np.min([prev_bet_fraction - (2 / (2 - np.log(3))) * (h / A_thing), 1/2]), -1/2])

                # update magnitude
                curr_magnitude = curr_bet_fraction * curr_wealth

                if len(all_weight_vectors) < 2:
                    final_w = weight_vector
                else:
                    final_w = np.mean(all_weight_vectors, axis=0)
                loss_thing = loss(x, y, final_w, loss_name='absolute')
                all_losses.append(loss_thing)

            print('initial wealth %5f | error %5.2f' % (initial_shit, loss_thing))
            print('')
            plt.plot(all_losses)
            plt.ylim(bottom=0, top=1)
            plt.pause(0.1)
            # update the final vector w
            self.w = np.mean(all_weight_vectors, axis=0)
            print(self.w)
        plt.show()

    def predict(self, x):
        w = self.w
        y_pred = np.sign(x @ w)
        return y_pred
