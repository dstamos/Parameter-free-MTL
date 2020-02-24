import numpy as np
import datetime
import matplotlib.pyplot as plt
from src.data_management import DataHandler, Settings
from src.indipendent_learning import ParameterFreeFixedBiasVariation
from src.parameter_free import ParameterFreeAggressiveVariation, ParameterFreeLazyVariation
import time
import argparse


def main():

    font = {'size': 26}
    import matplotlib
    matplotlib.rc('font', **font)

    all_all_errors_indi = []
    all_all_errors_oracle = []
    all_all_errors_lazy = []
    all_all_errors_aggressive = []

    tt = time.time()
    for seed in range(5):
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

        model = ParameterFreeFixedBiasVariation(np.zeros(settings.data.n_dims), verbose=1)
        all_errors_indi = model.fit(data, 'tr_task_indexes', linestyle='-')
        all_all_errors_indi.append(all_errors_indi)

        model = ParameterFreeFixedBiasVariation(10 * np.ones(settings.data.n_dims), verbose=1)
        all_errors_oracle = model.fit(data, 'tr_task_indexes', linestyle='--')
        all_all_errors_oracle.append(all_errors_oracle)

        model = ParameterFreeLazyVariation()
        all_errors_lazy = model.fit(data)
        all_all_errors_lazy.append(all_errors_lazy)

        model = ParameterFreeAggressiveVariation()
        all_errors_aggressive = model.fit(data)
        all_all_errors_aggressive.append(all_errors_aggressive)

        print('seed: %d | %5.2f sec' % (seed, time.time() - tt))

    indi_mean = np.mean(all_all_errors_indi, axis=0)
    indi_std = np.std(all_all_errors_indi, axis=0)

    oracle_mean = np.mean(all_all_errors_oracle, axis=0)
    oracle_std = np.std(all_all_errors_oracle, axis=0)

    lazy_mean = np.mean(all_all_errors_lazy, axis=0)
    lazy_std = np.std(all_all_errors_lazy, axis=0)

    aggro_mean = np.mean(all_all_errors_aggressive, axis=0)
    aggro_std = np.std(all_all_errors_aggressive, axis=0)

    my_dpi = 100
    plt.figure(figsize=(1280 / my_dpi, 720 / my_dpi), facecolor='white', dpi=my_dpi)

    plt.plot(indi_mean, color='k', linestyle='-', linewidth=2, label='ITL')
    plt.plot(oracle_mean, color='darkgreen', linestyle='--', linewidth=2,  label='Oracle')
    plt.plot(lazy_mean, color='tab:blue', linewidth=2, label='Lazy')
    plt.plot(aggro_mean, color='tab:red', linewidth=2, label='Aggressive')
    plt.ylim(bottom=2, top=12)

    plt.fill_between(range(len(indi_mean)), indi_mean - indi_std, indi_mean + indi_std,
                     alpha=0.1, edgecolor='k', facecolor='k', antialiased=True)

    plt.fill_between(range(len(oracle_mean)), oracle_mean - oracle_std, oracle_mean + oracle_std,
                     alpha=0.1, edgecolor='darkgreen', facecolor='darkgreen', antialiased=True)

    plt.fill_between(range(len(lazy_mean)), lazy_mean - lazy_std, lazy_mean + lazy_std,
                     alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True)

    plt.fill_between(range(len(aggro_mean)), aggro_mean - aggro_std, aggro_mean + aggro_std,
                     alpha=0.1, edgecolor='tab:red', facecolor='tab:red', antialiased=True)

    plt.legend()
    plt.xlabel('iterations', fontsize=38, fontweight="normal")
    plt.ylabel('cumulative error', fontsize=38, fontweight="normal")

    plt.tight_layout()
    plt.savefig('temp ' + str(datetime.datetime.now()).replace(':', '') + '.png', format='png')
    plt.pause(0.01)
    plt.close()

    exit()


if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description="Parameter-free metalearning")
    # argparser.add_argument('--within_task_step_size', type=float, help='Within task step size', default=0.1)
    # argparser.add_argument('--within_task_radius', type=float, help='Within task radius', default=0.5)
    # argparser.add_argument('--outer_task_step_size', type=float, help='Outer task step size', default=0.3)
    # args = argparser.parse_args()

    main()
