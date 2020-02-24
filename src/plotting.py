import numpy as np
import matplotlib.pyplot as plt
import datetime


def plot_stuff(indi, oracle, aggressive, plot_type='cumulative_errors', lazy=None):

    indi_mean = np.mean(indi, axis=0)
    indi_std = np.std(indi, axis=0)

    oracle_mean = np.mean(oracle, axis=0)
    oracle_std = np.std(oracle, axis=0)

    if plot_type == 'cumulative_errors':
        lazy_mean = np.mean(lazy, axis=0)
        lazy_std = np.std(lazy, axis=0)

    aggro_mean = np.mean(aggressive, axis=0)
    aggro_std = np.std(aggressive, axis=0)

    my_dpi = 100
    plt.figure(figsize=(1280 / my_dpi, 720 / my_dpi), facecolor='white', dpi=my_dpi)

    plt.plot(indi_mean, color='k', linestyle='-', linewidth=2, label='ITL')
    plt.plot(oracle_mean, color='darkgreen', linestyle='--', linewidth=2, label='Oracle')
    if plot_type == 'cumulative_errors':
        plt.plot(lazy_mean, color='tab:blue', linewidth=2, label='Lazy')
    plt.plot(aggro_mean, color='tab:red', linewidth=2, label='Aggressive')
    plt.ylim(bottom=2, top=12)

    plt.fill_between(range(len(indi_mean)), indi_mean - indi_std, indi_mean + indi_std,
                     alpha=0.1, edgecolor='k', facecolor='k', antialiased=True)

    plt.fill_between(range(len(oracle_mean)), oracle_mean - oracle_std, oracle_mean + oracle_std,
                     alpha=0.1, edgecolor='darkgreen', facecolor='darkgreen', antialiased=True)

    if plot_type == 'cumulative_errors':
        plt.fill_between(range(len(lazy_mean)), lazy_mean - lazy_std, lazy_mean + lazy_std,
                         alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True)

    plt.fill_between(range(len(aggro_mean)), aggro_mean - aggro_std, aggro_mean + aggro_std,
                     alpha=0.1, edgecolor='tab:red', facecolor='tab:red', antialiased=True)

    plt.legend()
    if plot_type == 'cumulative_errors':
        plt.xlabel('iterations', fontsize=38, fontweight="normal")
        plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
    else:
        plt.xlabel('number of tasks', fontsize=38, fontweight="normal")
        plt.ylabel('test error', fontsize=38, fontweight="normal")

    plt.tight_layout()
    plt.savefig('temp_' + plot_type + '_' + str(datetime.datetime.now()).replace(':', '') + '.png', format='png')
    plt.pause(0.01)
    plt.close()
