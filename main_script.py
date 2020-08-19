import numpy as np
from src.data_management import DataHandler, Settings
from src.indipendent_learning import ParameterFreeFixedBiasVariation
from src.parameter_free_aggressive_mtl import ParameterFreeAggressiveVariation, ParameterFreeAggressiveClassic
from src.parameter_free_lazy_mtl import ParameterFreeLazyVariation, ParameterFreeLazyClassic
from src.step_size_tuning import ParameterFreeLazyVariationStepSearch, ParameterFreeAggressiveVariationStepSearch
import time
from src.plotting import plot_stuff


def main():

    font = {'size': 26}
    import matplotlib
    matplotlib.rc('font', **font)

    import pickle

    a = pickle.load(open('presaved_50d_picklefile.dat', "rb"))

    # methods = ['ITL', 'Oracle', 'Aggressive', 'Lazy', 'Aggressive_KT', 'Lazy_KT', 'Aggressive_SS', 'Lazy_SS']
    # methods = ['ITL', 'Oracle', 'Aggressive', 'Lazy', 'Aggressive_KT', 'Lazy_KT']
    # methods = ['ITL', 'Oracle', 'Aggressive', 'Lazy']

    # methods = ['ITL', 'Aggressive', 'Lazy', 'Aggressive_KT', 'Lazy_KT', 'Aggressive_SS', 'Lazy_SS']
    methods = ['ITL', 'Oracle', 'Aggressive', 'Lazy', 'Aggressive_KT', 'Lazy_KT']
    # methods = ['ITL', 'Aggressive', 'Lazy']

    results = {}
    for curr_method in methods:
        results[curr_method + '_mtl'] = []
        results[curr_method + '_accu'] = []

    tt = time.time()
    for seed in range(30):
        np.random.seed(seed)
        general_settings = {'seed': seed,
                            'verbose': 1}

        data_settings = {'dataset': 'synthetic-regression',
                         'n_tr_tasks': 400,
                         'n_val_tasks': 20,
                         'n_test_tasks': 50,
                         'n_all_points': 50,
                         'ts_points_pct': 0.5,
                         'n_dims': 10,
                         'noise_std': 1}

        # data_settings = {'dataset': 'synthetic-regression',
        #                  'n_tr_tasks': 400,
        #                  'n_val_tasks': 20,
        #                  'n_test_tasks': 50,
        #                  'n_all_points': 10,
        #                  'ts_points_pct': 0.5,
        #                  'n_dims': 5,
        #                  'noise_std': 1}

        # data_settings = {'dataset': 'schools',
        #                  'n_tr_tasks': 139,
        #                  'n_val_tasks': 0,
        #                  'n_test_tasks': 0,
        #                  'ts_points_pct': 0.25
        #                  }

        # data_settings = {'dataset': 'lenk',
        #                  'n_tr_tasks': 180,
        #                  'n_val_tasks': 0,
        #                  'n_test_tasks': 0,
        #                  'ts_points_pct': 0.20
        #                  }

        settings = Settings(data_settings, 'data')
        settings.add_settings(general_settings)

        data = DataHandler(settings)

        for curr_method in methods:
            print(curr_method)
            if curr_method == 'ITL':
                model = ParameterFreeFixedBiasVariation(np.zeros(data.features_tr[0].shape[1]))
                mtl_errors, accumulated_errors = model.fit(data, 'tr_task_indexes')
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Oracle':
                model = ParameterFreeFixedBiasVariation(data.oracle)
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
            elif curr_method == 'Aggressive_KT':
                model = ParameterFreeAggressiveClassic()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Lazy_KT':
                model = ParameterFreeLazyClassic()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Aggressive_SS':
                model = ParameterFreeAggressiveVariationStepSearch()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            elif curr_method == 'Lazy_SS':
                model = ParameterFreeLazyVariationStepSearch()
                mtl_errors, accumulated_errors = model.fit(data)
                results[curr_method + '_mtl'].append(mtl_errors)
                results[curr_method + '_accu'].append(accumulated_errors)
            print('%s done %5.2f' % (curr_method, time.time() - tt))
        print('seed: %d | %5.2f sec' % (seed, time.time() - tt))

        import os
        import pickle
        if not os.path.exists('results'):
            os.makedirs('results')
        f = open('results' + '/' + settings.data.dataset + ".pckl", 'wb')
        pickle.dump(results, f)
        f.close()

    plot_stuff(results, methods, settings.data.dataset)

    # results = pickle.load(open('results/' + str(settings.data.dataset) + '.pckl', "rb"))

    # exit()


if __name__ == "__main__":

    main()
#