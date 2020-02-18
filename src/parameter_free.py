import numpy as np


class ParameterFreeFixedBias:
    def __init__(self, fixed_bias, initial_wealth, lipschitz_constant, input_norm_bound):
        self.lipschitz_constant = lipschitz_constant
        self.input_norm_bound = input_norm_bound
        self.fixed_bias = fixed_bias
        self.w = None
        self.magnitude_betting_fraction = 0
        self.magnitude_wealth = initial_wealth

    def fit(self, x, y):
        n_points, n_dims = x.shape
        curr_magnitude_betting_fraction = self.magnitude_betting_fraction
        curr_magnitude_wealth = self.magnitude_wealth
        curr_magnitude = curr_magnitude_betting_fraction * curr_magnitude_wealth
        curr_direction = np.zeros(n_dims)

        all_weight_vectors = []
        shuffled_indexes = list(range(n_points))
        np.random.shuffle(shuffled_indexes)
        for iteration, curr_point_idx in enumerate(shuffled_indexes):
            iteration = iteration + 1
            prev_direction = curr_direction
            prev_magnitude_betting_fraction = curr_magnitude_betting_fraction
            prev_magnitude_wealth = curr_magnitude_wealth
            prev_magnitude = curr_magnitude

            # update weight vector
            weight_vector = prev_magnitude * prev_direction + self.fixed_bias
            all_weight_vectors.append(weight_vector)

            # receive a new datapoint
            curr_x = x[curr_point_idx, :]
            curr_y = y[curr_point_idx]

            # compute the gradient
            subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='hinge')
            full_gradient = subgrad * curr_x

            # define step size
            step_size = np.sqrt(2 / (self.lipschitz_constant * self.input_norm_bound * iteration))

            # update direction
            curr_direction = l2_unit_ball_projection(prev_direction - step_size * full_gradient)

            # update magnitude_wealth
            curr_magnitude_wealth = prev_magnitude_wealth - 1 / (self.input_norm_bound * self.lipschitz_constant) * full_gradient @ prev_direction * prev_magnitude

            # update magnitude_betting_fraction
            curr_magnitude_betting_fraction = -(1 / iteration)*((iteration - 1) * prev_magnitude_betting_fraction + 1 / (self.lipschitz_constant * self.input_norm_bound) * full_gradient @ prev_direction)

            # update magnitude
            curr_magnitude = curr_magnitude_betting_fraction * curr_magnitude_wealth

            # FIXME
            # print(weight_vector)
            if len(all_weight_vectors) < 2:
                final_w = weight_vector
            else:
                final_w = np.mean(all_weight_vectors, axis=0)
            print('%4d | total loss: %6.3f' % (iteration, loss(x, y, final_w)))
        # update the final vector w
        self.w = np.mean(all_weight_vectors, axis=0)
        print(self.w)

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
                meta_step_size = np.sqrt(2 / (self.lipschitz_constant * self.input_norm_bound * total_iter))

                # update meta-direction
                curr_meta_direction = l2_unit_ball_projection(prev_meta_direction - meta_step_size * full_gradient)

                # define inner step size
                inner_step_size = np.sqrt(2 / (self.lipschitz_constant * self.input_norm_bound * inner_iteration))

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

            self.all_weight_vectors.append(np.mean(temp_weight_vectors))
        self.all_meta_parameters = all_meta_parameters

    @staticmethod
    def __general_iteration(t, n, i):
        return (t - 1) * n + i

    @staticmethod
    def predict(x, w):
        # w = self.w
        y_pred = np.sign(x @ w)
        return y_pred


class ParameterFreeLazy:
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

        total_iter = 0
        all_meta_parameters = []
        for task_iteration, (x, y) in enumerate(zip(x_list, y_list)):
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

            temp_weight_vectors = []
            all_gradients = []
            shuffled_indexes = list(range(n_points))
            np.random.shuffle(shuffled_indexes)
            for inner_iteration, curr_point_idx in enumerate(shuffled_indexes):
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
                subgrad = subgradient(curr_x, curr_y, weight_vector, loss_name='hinge')
                full_gradient = subgrad * curr_x
                all_gradients.append(full_gradient)

                # define inner step size
                inner_step_size = np.sqrt(2 / (self.lipschitz_constant * self.input_norm_bound * inner_iteration))

                # update inner direction
                curr_inner_direction = l2_unit_ball_projection(prev_inner_direction - inner_step_size * full_gradient)

                # update inner magnitude_wealth
                curr_inner_magnitude_wealth = prev_inner_magnitude_wealth - 1 / (self.input_norm_bound * self.lipschitz_constant) * full_gradient @ prev_inner_direction * prev_inner_magnitude

                # update magnitude_betting_fraction
                curr_inner_magnitude_betting_fraction = -(1 / inner_iteration) * ((inner_iteration - 1) * prev_inner_magnitude_betting_fraction + 1 / (self.lipschitz_constant * self.input_norm_bound) * full_gradient @ prev_inner_direction)

                # update magnitude
                curr_inner_magnitude = curr_inner_magnitude_betting_fraction * curr_inner_magnitude_wealth

                print('%4d | magnitude: %6.3f' % (total_iter, curr_inner_magnitude))

            # define total iteration
            total_iter = total_iter + n_points

            # compute meta-gradient
            meta_gradient = np.sum(all_gradients)

            # define meta step size
            meta_step_size = np.sqrt(2 / (self.lipschitz_constant * self.input_norm_bound * total_iter))

            # update meta-direction
            curr_meta_direction = l2_unit_ball_projection(prev_meta_direction - meta_step_size * meta_gradient)

            # update meta-magnitude_wealth
            curr_meta_magnitude_wealth = prev_meta_magnitude_wealth - 1 / (self.input_norm_bound * self.lipschitz_constant) * meta_gradient @ prev_meta_direction * prev_meta_magnitude

            # update meta-magnitude_betting_fraction
            curr_meta_magnitude_betting_fraction = - (1/total_iter) * ((total_iter - 1) * prev_meta_magnitude_betting_fraction + 1 / (self.lipschitz_constant * self.input_norm_bound) * meta_gradient @ prev_meta_direction)

            # update meta-magnitude
            curr_meta_magnitude = curr_meta_magnitude_betting_fraction * curr_meta_magnitude_wealth

            self.all_weight_vectors.append(temp_weight_vectors)
        self.all_meta_parameters = all_meta_parameters

    @staticmethod
    def __general_iteration(t, n, i):
        return (t - 1) * n + i

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
            return -y * x
        else:
            return 0
    else:
        raise ValueError("Unknown loss.")


def loss(x, y, w, loss_name='hinge'):
    from sklearn.metrics import hinge_loss
    try:
        return hinge_loss(y, x @ w)
    except:
        k = 1
