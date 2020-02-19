import numpy as np
from src.data_management import DataHandler, Settings
from src.indipendent_learning import BasicBias, ParameterFreeFixedBiasVariation, ParameterFreeFixedBias
from src.parameter_free import ParameterFreeAggressiveVariation, ParameterFreeLazyVariation
import argparse


def main():

    seed = 999
    np.random.seed(seed)
    general_settings = {'seed': seed,
                        'verbose': 1}

    data_settings = {'dataset': 'synthetic',
                     'n_tr_tasks': 2,
                     'n_val_tasks': 2,
                     'n_test_tasks': 5,
                     'n_all_points': 500,
                     'ts_points_pct': 0.5,
                     'n_dims': 2,
                     'noise_std': 0.1}

    training_settings = {'method': 'dynamic-metal_ml',
                         'within_task_step_size': 0.1,
                         'within_task_radius': 0.5,
                         'outer_task_step_size': 0.3}

    settings = Settings(data_settings, 'data')
    settings.add_settings(training_settings, 'training')
    settings.add_settings(general_settings)

    data = DataHandler(settings)

    # model = BasicBias(np.zeros(settings.data.n_dims))
    # model.fit(data.features_tr[1], data.labels_tr[1])

    model = ParameterFreeFixedBiasVariation(0 * np.ones(settings.data.n_dims), 1, 1, 1)
    model.fit(data.features_tr[0], data.labels_tr[0])

    # model = ParameterFreeAggressiveVariation(1, 1, 1, 1)
    # model.fit(data.features_tr, data.labels_tr)
    # k = 1


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description="Parameter-free metalearning")
    # argparser.add_argument('--within_task_step_size', type=float, help='Within task step size', default=0.1)
    # argparser.add_argument('--within_task_radius', type=float, help='Within task radius', default=0.5)
    # argparser.add_argument('--outer_task_step_size', type=float, help='Outer task step size', default=0.3)
    # args = argparser.parse_args()

    main()
