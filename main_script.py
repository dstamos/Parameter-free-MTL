import numpy as np
from src.data_management import DataHandler, Settings
from src.indipendent_learning import ParameterFreeFixedBiasVariation
from src.parameter_free import ParameterFreeAggressiveVariation, ParameterFreeLazyVariation, ParameterFreeAggressiveClassic, ParameterFreeLazyClassic
from src.step_size_tuning import ParameterFreeLazyVariationStepSearch, ParameterFreeAggressiveVariationStepSearch
import time
from src.plotting import plot_stuff
import argparse


def main():

    font = {'size': 26}
    import matplotlib
    matplotlib.rc('font', **font)

    all_accumulative_errors_indi = []
    all_mtl_errors_indi = []
    all_accumulative_errors_oracle = []
    all_mtl_errors_oracle = []
    all_errors_lazy = []
    all_accumulative_errors_aggressive = []
    all_mtl_errors_aggressive = []

    tt = time.time()
    for seed in range(2):
        np.random.seed(seed)
        general_settings = {'seed': seed,
                            'verbose': 1}

        data_settings = {'dataset': 'synthetic-regression',
                         'n_tr_tasks': 400,
                         'n_val_tasks': 20,
                         'n_test_tasks': 50,
                         'n_all_points': 40,
                         'ts_points_pct': 0.5,
                         'n_dims': 30,
                         'noise_std': 0.1}

        training_settings = {'method': 'dynamic-metal_ml',
                             'within_task_step_size': 0.1,
                             'within_task_radius': 0.5,
                             'outer_task_step_size': 0.3}

        settings = Settings(data_settings, 'data')
        settings.add_settings(training_settings, 'training')
        settings.add_settings(general_settings)

        data = DataHandler(settings)

        model = ParameterFreeFixedBiasVariation(np.zeros(data.oracle.shape), verbose=1)
        mtl_errors_indi, accumulative_errors_indi = model.fit(data, 'tr_task_indexes')
        all_accumulative_errors_indi.append(accumulative_errors_indi)
        all_mtl_errors_indi.append(mtl_errors_indi)

        model = ParameterFreeFixedBiasVariation(data.oracle, verbose=1)
        mtl_errors_oracle, accumulative_errors_oracle = model.fit(data, 'tr_task_indexes')
        all_accumulative_errors_oracle.append(accumulative_errors_oracle)
        all_mtl_errors_oracle.append(mtl_errors_oracle)

        model = ParameterFreeLazyVariation()
        errors_lazy = model.fit(data)
        all_errors_lazy.append(errors_lazy)

        model = ParameterFreeAggressiveVariation()
        mtl_errors_aggressive, cumulative_errors_aggressive = model.fit(data)
        all_accumulative_errors_aggressive.append(cumulative_errors_aggressive)
        all_mtl_errors_aggressive.append(mtl_errors_aggressive)

        print('seed: %d | %5.2f sec' % (seed, time.time() - tt))

    plot_stuff(all_accumulative_errors_indi, all_accumulative_errors_oracle, all_accumulative_errors_aggressive, lazy=all_errors_lazy, plot_type='cumulative_errors')
    plot_stuff(all_mtl_errors_indi, all_mtl_errors_oracle, all_mtl_errors_aggressive, plot_type='mtl_errors')

    exit()


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description="Parameter-free metalearning")
    # argparser.add_argument('--within_task_step_size', type=float, help='Within task step size', default=0.1)
    # argparser.add_argument('--within_task_radius', type=float, help='Within task radius', default=0.5)
    # argparser.add_argument('--outer_task_step_size', type=float, help='Outer task step size', default=0.3)
    # args = argparser.parse_args()

    main()
