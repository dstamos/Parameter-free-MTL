import numpy as np
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


class ParameterFreeAggressiveVariation:
    def __init__(self, meta_initial_wealth, inner_initial_wealth, lipschitz_constant, input_norm_bound):
        self.L = lipschitz_constant
        self.R = input_norm_bound
        self.all_weight_vectors = []
        self.all_meta_parameters = []
        self.meta_magnitude_betting_fraction = 0
        self.meta_magnitude_wealth = meta_initial_wealth
        self.inner_magnitude_betting_fraction = 0
        self.inner_magnitude_wealth = inner_initial_wealth

    def fit(self, x_list, y_list):
        n_dims = x_list[0].shape[1]

        # FIXME
        n_points = x_list[0].shape[0]
        # range_shit = np.linspace(0.1, 1000, 50)
        # range_shit = np.linspace(0.1, n_points * len(x_list), 30)
        range_shit_inner = np.linspace(0.1, n_points, 6)
        range_shit_meta = np.linspace(0.1, n_points * len(x_list), 6)
        # range_shit = [1000]
        for idx, value_shit_meta in enumerate(range_shit_meta):
            for idx, value_shit_inner in enumerate(range_shit_inner):
                print(value_shit_meta, value_shit_inner)
                curr_meta_fraction = self.meta_magnitude_betting_fraction
                curr_meta_wealth = value_shit_meta  # self.meta_magnitude_wealth
                curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth
                curr_meta_direction = np.zeros(n_dims)

                all_losses = []
                all_h_meta = []
                manual_rolling_average = []
                total_sum = 0
                all_meta_parameters = []
                for task_iteration, (x, y) in enumerate(zip(x_list, y_list)):
                    task_iteration = task_iteration + 1
                    prev_meta_direction = curr_meta_direction
                    prev_meta_fraction = curr_meta_fraction
                    prev_meta_wealth = value_shit_inner  # curr_meta_wealth
                    prev_meta_magnitude = curr_meta_magnitude

                    # initialize the inner parameters
                    n_points, n_dims = x.shape
                    curr_inner_fraction = self.inner_magnitude_betting_fraction
                    curr_inner_wealth = self.inner_magnitude_wealth
                    curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth
                    curr_inner_direction = np.zeros(n_dims)

                    all_h_inner = []

                    temp_weight_vectors = []
                    shuffled_indexes = list(range(n_points))
                    np.random.shuffle(shuffled_indexes)
                    for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                        inner_iteration = inner_iteration + 1
                        prev_inner_direction = curr_inner_direction
                        prev_inner_fraction = curr_inner_fraction
                        prev_inner_wealth = curr_inner_wealth
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
                        subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
                        full_gradient = subgrad * curr_x

                        # define meta step size
                        meta_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / total_iter)

                        # update meta-direction
                        curr_meta_direction = l2_unit_ball_projection(prev_meta_direction - meta_step_size * full_gradient)

                        # define inner step size
                        inner_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / inner_iteration)

                        # update inner direction
                        curr_inner_direction = l2_unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

                        # update meta-magnitude_wealth
                        curr_meta_wealth = prev_meta_wealth - (1 / (self.R * self.L)) * full_gradient @ prev_meta_direction * prev_meta_magnitude

                        # meta h thing
                        h_meta = (1 / (self.R * self.L)) * (full_gradient @ prev_meta_direction) * (1 / (1 - (1 / (self.R * self.L)) * (full_gradient @ prev_meta_direction) * prev_meta_fraction))
                        all_h_meta.append(h_meta)
                        A_thing_meta = 1 + np.sum([curr_h ** 2 for curr_h in all_h_meta])

                        # update meta-magnitude_betting_fraction
                        curr_meta_fraction = np.max([np.min([prev_meta_fraction - (2 / (2 - np.log(3))) * (h_meta / A_thing_meta), 1 / 2]), -1 / 2])

                        # update meta-magnitude
                        curr_meta_magnitude = curr_meta_fraction * curr_meta_wealth

                        # update inner magnitude_wealth
                        curr_inner_wealth = prev_inner_wealth - 1 / (self.R * self.L) * full_gradient @ prev_inner_direction * prev_inner_magnitude

                        # update magnitude_betting_fraction
                        h_inner = (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction * (1 / (1 - (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction * prev_inner_fraction))
                        all_h_inner.append(h_inner)
                        A_thing_inner = 1 + np.sum([curr_h**2 for curr_h in all_h_inner])
                        # update magnitude_betting_fraction
                        curr_inner_fraction = np.max([np.min([prev_inner_fraction - (2 / (2 - np.log(3))) * (h_inner / A_thing_inner), 1/2]), -1/2])

                        # update magnitude
                        curr_inner_magnitude = curr_inner_fraction * curr_inner_wealth

                        loss_thing = loss(curr_x, curr_y, np.mean(temp_weight_vectors, axis=0), loss_name='absolute')
                        all_losses.append(loss_thing)

                        total_sum = total_sum + loss_thing
                        manual_rolling_average.append(total_sum / total_iter)
                    plt.plot(manual_rolling_average)
                    plt.ylim(bottom=0, top=2)
                    plt.title([str(value_shit_meta) + ' ' + str(value_shit_inner)])
                    plt.pause(0.1)

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


class ParameterFreeLazyVariation:
    def __init__(self, meta_initial_wealth, inner_initial_wealth, lipschitz_constant, input_norm_bound):
        self.L = lipschitz_constant
        self.R = input_norm_bound
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

        all_h_meta = []
        all_losses = []
        total_sum = 0
        temp_iter = 0
        manual_rolling_average = []

        total_iter = 0
        all_meta_parameters = []
        for task_iteration, (x, y) in enumerate(zip(x_list, y_list)):
            task_iteration = task_iteration + 1
            prev_meta_direction = curr_meta_direction
            prev_meta_magnitude_betting_fraction = curr_meta_magnitude_betting_fraction
            prev_meta_magnitude_wealth = curr_meta_magnitude_wealth
            prev_meta_magnitude = curr_meta_magnitude

            # update meta-parameter
            meta_parameter = prev_meta_magnitude * prev_meta_direction
            all_meta_parameters.append(meta_parameter)

            # initialize the inner parameters
            n_points, n_dims = x.shape
            curr_inner_magnitude_betting_fraction = self.inner_magnitude_betting_fraction
            curr_inner_magnitude_wealth = self.inner_magnitude_wealth
            curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth
            curr_inner_direction = np.zeros(n_dims)

            all_h_inner = []

            temp_weight_vectors = []
            all_gradients = []
            shuffled_indexes = list(range(n_points))
            np.random.shuffle(shuffled_indexes)
            for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
                inner_iteration = inner_iteration + 1
                prev_inner_direction = curr_inner_direction
                prev_inner_magnitude_betting_fraction = curr_inner_magnitude_betting_fraction
                prev_inner_magnitude_wealth = curr_inner_magnitude_wealth
                prev_inner_magnitude = curr_inner_magnitude

                # update inner weight vector
                weight_vector = prev_inner_magnitude * prev_inner_direction + meta_parameter
                temp_weight_vectors.append(weight_vector)

                # receive a new datapoint
                curr_x = x[curr_point_idx, :]
                curr_y = y[curr_point_idx]

                # compute the gradient
                subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='absolute')
                full_gradient = subgrad * curr_x
                all_gradients.append(full_gradient)

                # define inner step size
                inner_step_size = (1 / (self.L * self.R)) * np.sqrt(2 / inner_iteration)

                # update inner direction
                curr_inner_direction = l2_unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

                # update inner magnitude_wealth
                curr_inner_magnitude_wealth = prev_inner_magnitude_wealth - 1 / (self.R * self.L) * full_gradient @ prev_inner_direction * prev_inner_magnitude

                # update magnitude_betting_fraction
                h_inner = (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction * (1 / (1 - (1 / (self.R * self.L)) * full_gradient @ prev_inner_direction * prev_inner_magnitude_betting_fraction))
                all_h_inner.append(h_inner)
                A_thing_inner = 1 + np.sum([curr_h ** 2 for curr_h in all_h_inner])

                curr_inner_magnitude_betting_fraction = np.max([np.min([prev_inner_magnitude_betting_fraction - (2 / (2 - np.log(3))) * (h_inner / A_thing_inner), 1 / 2]), -1 / 2])

                # update magnitude
                curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth

                loss_thing = loss(curr_x, curr_y, np.mean(temp_weight_vectors, axis=0), loss_name='absolute')
                all_losses.append(loss_thing)

                total_sum = total_sum + loss_thing
                temp_iter = temp_iter + 1
                manual_rolling_average.append(total_sum / temp_iter)

            # define total iteration
            total_iter = total_iter + n_points

            # compute meta-gradient
            meta_gradient = np.sum(all_gradients, axis=0)

            # define meta step size
            meta_step_size = (1 / (self.L * self.R * n_points)) * np.sqrt(2 / task_iteration)

            # update meta-direction
            curr_meta_direction = l2_unit_ball_projection(prev_meta_direction - meta_step_size * meta_gradient)

            # update meta-magnitude_wealth
            curr_meta_magnitude_wealth = prev_meta_magnitude_wealth - (1 / (self.R * self.L * n_points)) * meta_gradient @ prev_meta_direction * prev_meta_magnitude

            # update meta-magnitude_betting_fraction
            h_meta = (1 / (self.R * self.L * n_points)) * (meta_gradient @ prev_meta_direction) * (1 / (1 - (1 / (self.R * self.L * n_points)) * (meta_gradient @ prev_meta_direction) * prev_meta_magnitude_betting_fraction))
            all_h_meta.append(h_meta)
            A_thing_meta = 1 + np.sum([curr_h ** 2 for curr_h in all_h_meta])

            curr_meta_magnitude_betting_fraction = np.max([np.min([prev_meta_magnitude_betting_fraction - (2 / (2 - np.log(3))) * (h_meta / A_thing_meta), 1/2]), -1/2])

            # update meta-magnitude
            curr_meta_magnitude = curr_meta_magnitude_betting_fraction * curr_meta_magnitude_wealth

            plt.plot(manual_rolling_average)
            plt.ylim(bottom=0, top=2)
            plt.pause(0.1)

        self.all_meta_parameters = all_meta_parameters

    @staticmethod
    def predict(x, w):
        # w = self.w
        y_pred = np.sign(x @ w)
        return y_pred


def l2_unit_ball_projection(vector):
    return vector / np.maximum(1, np.linalg.norm(vector, ord=2))


def subgradient(x, y, w, loss_name=None):
    if loss_name == 'hinge':
        score = x @ w
        if y * score < 1:
            return -y
        else:
            return 0
    elif loss_name == 'absolute':
        pred = x @ w
        if y < pred:
            return 1
        elif y >= pred:
            return -1
    else:
        raise ValueError("Unknown loss.")


def loss(x, y, w, loss_name='hinge'):
    if loss_name == 'hinge':
        from sklearn.metrics import hinge_loss
        loss_thing = hinge_loss(y, x @ w)
        return loss_thing
    elif loss_name == 'absolute':
        if hasattr(y, '__len__'):
            return 1 / len(y) * np.sum(np.abs(y - x @ w))
        else:
            return 1 / len(y) * np.sum(np.abs(y - x @ w))
    else:
        raise ValueError("Unknown loss.")
