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

    methods = ['ITL', 'Oracle', 'Aggressive', 'Lazy', 'Aggressive_classic', 'Lazy_classic', 'Aggressive_step_search', 'Lazy_step_search']
    results = {}
    for curr_method in methods:
        results[curr_method + '_mtl'] = []
        results[curr_method + '_accu'] = []

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

        for curr_method in methods:
            if curr_method == 'ITL':
                model = ParameterFreeFixedBiasVariation(np.zeros(data.oracle.shape), verbose=1)
                mtl_errors, accumulated_errors = model.fit(data, 'tr_task_indexes')
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Oracle':
                model = ParameterFreeFixedBiasVariation(data.oracle, verbose=1)
                mtl_errors, accumulated_errors = model.fit(data, 'tr_task_indexes')
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Aggressive':
                model = ParameterFreeAggressiveVariation()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Lazy':
                model = ParameterFreeLazyVariation()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Aggressive_classic':
                model = ParameterFreeAggressiveClassic()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Lazy_classic':
                model = ParameterFreeLazyClassic()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Aggressive_step_search':
                model = ParameterFreeAggressiveVariationStepSearch()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Lazy_step_search':
                model = ParameterFreeLazyVariationStepSearch()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)

        print('seed: %d | %5.2f sec' % (seed, time.time() - tt))

    plot_stuff(results, methods)

    exit()


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description="Parameter-free metalearning")
    # argparser.add_argument('--within_task_step_size', type=float, help='Within task step size', default=0.1)
    # argparser.add_argument('--within_task_radius', type=float, help='Within task radius', default=0.5)
    # argparser.add_argument('--outer_task_step_size', type=float, help='Outer task step size', default=0.3)
    # args = argparser.parse_args()

    main()
