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


class ParameterFreeFixedBias:
    def __init__(self, fixed_bias, initial_wealth=1, lipschitz_constant=1, input_norm_bound=1):
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
