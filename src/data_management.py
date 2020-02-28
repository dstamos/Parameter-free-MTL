import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split


class Settings:
    def __init__(self, dictionary, struct_name=None):
        if struct_name is None:
            self.__dict__.update(**dictionary)
        else:
            temp_settings = Settings(dictionary)
            setattr(self, struct_name, temp_settings)

    def add_settings(self, dictionary, struct_name=None):
        if struct_name is None:
            self.__dict__.update(dictionary)
        else:
            if hasattr(self, struct_name):
                temp_settings = getattr(self, struct_name)
                temp_settings.__dict__.update(dictionary)
            else:
                temp_settings = Settings(dictionary)
            setattr(self, struct_name, temp_settings)


class DataHandler:
    def __init__(self, settings):
        settings.add_settings({'n_all_tasks': settings.data.n_tr_tasks + settings.data.n_val_tasks + settings.data.n_test_tasks}, 'data')
        self.settings = settings
        self.features_tr = [None] * settings.data.n_all_tasks
        self.features_ts = [None] * settings.data.n_all_tasks
        self.labels_tr = [None] * settings.data.n_all_tasks
        self.labels_ts = [None] * settings.data.n_all_tasks
        self.oracle = None

        self.tr_task_indexes = None
        self.val_task_indexes = None
        self.test_task_indexes = None

        if self.settings.data.dataset == 'synthetic-regression':
            self.synthetic_regression_data_gen()
        elif self.settings.data.dataset == 'synthetic-classification':
            self.synthetic_classification_data_gen()
        else:
            raise ValueError('Invalid dataset')

    def synthetic_regression_data_gen(self):
        self.oracle = 10 * np.ones(self.settings.data.n_dims)
        for task_idx in range(self.settings.data.n_all_tasks):
            # generating and normalizing the inputs
            features = np.random.randn(self.settings.data.n_all_points, self.settings.data.n_dims)
            features = features / norm(features, axis=1, keepdims=True)

            # generating and normalizing the weight vectors
            weight_vector = self.oracle + np.random.normal(loc=np.zeros(self.settings.data.n_dims), scale=1).ravel()

            # generating labels and adding noise
            clean_labels = features @ weight_vector

            signal_to_noise_ratio = 10
            standard_noise = np.random.randn(self.settings.data.n_all_points)
            noise_std = np.sqrt(np.var(clean_labels) / (signal_to_noise_ratio * np.var(standard_noise)))
            noisy_labels = clean_labels + noise_std * standard_noise

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.settings.data.n_all_points), test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]

            features_ts = features[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks, self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks, self.settings.data.n_all_tasks)

    def synthetic_classification_data_gen(self):
        self.oracle = 10 * np.ones(self.settings.data.n_dims)
        for task_idx in range(self.settings.data.n_all_tasks):
            # generating and normalizing the inputs
            features = np.random.randn(self.settings.data.n_all_points, self.settings.data.n_dims)
            features = features / norm(features, axis=1, keepdims=True)

            # generating and normalizing the weight vectors
            weight_vector = self.oracle + np.random.normal(loc=np.zeros(self.settings.data.n_dims), scale=1).ravel()

            # generating labels and adding noise
            clean_scores = features @ weight_vector

            noisy_labels = -1 * np.ones(len(clean_scores))
            noisy_labels[(1 / (1 + 10 * np.exp(-clean_scores))) > 0.5] = 1

            # split into training and test
            tr_indexes, ts_indexes = train_test_split(np.arange(0, self.settings.data.n_all_points), test_size=self.settings.data.ts_points_pct)
            features_tr = features[tr_indexes]
            labels_tr = noisy_labels[tr_indexes]

            features_ts = features[ts_indexes]
            labels_ts = noisy_labels[ts_indexes]

            self.features_tr[task_idx] = features_tr
            self.features_ts[task_idx] = features_ts
            self.labels_tr[task_idx] = labels_tr
            self.labels_ts[task_idx] = labels_ts

        self.tr_task_indexes = np.arange(0, self.settings.data.n_tr_tasks)
        self.val_task_indexes = np.arange(self.settings.data.n_tr_tasks, self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks)
        self.test_task_indexes = np.arange(self.settings.data.n_tr_tasks + self.settings.data.n_val_tasks, self.settings.data.n_all_tasks)
