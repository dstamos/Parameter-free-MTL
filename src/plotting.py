import numpy as np
import matplotlib.pyplot as plt
import datetime


def plot_stuff(results, methods):

    linestyles = ['-', ':', '--', '-.']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black',
              'crimson', 'red', 'green', 'blue']

    my_dpi = 100
    plt.figure(figsize=(1664 / my_dpi, 936 / my_dpi), facecolor='white', dpi=my_dpi)

    plt.ylim(bottom=2, top=10)
    for idx, curr_method in enumerate(methods):
        color = colors[idx]
        linestyle = np.random.choice(linestyles, 1)[0]

        mean = np.nanmean(results[curr_method + '_accu'], axis=0)
        std = np.nanstd(results[curr_method + '_accu'], axis=0)

        plt.plot(mean, color=color, linestyle=linestyle, linewidth=2, label=curr_method)

        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1, edgecolor=color, facecolor=color, antialiased=True)
        plt.xlabel('iterations', fontsize=38, fontweight="normal")
        plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
        plt.legend()

    plt.tight_layout()
    plt.savefig('temp_accu' + '_' + str(datetime.datetime.now()).replace(':', '') + '.png', format='png')
    plt.pause(0.01)
    plt.close()

    my_dpi = 100
    plt.figure(figsize=(1664 / my_dpi, 936 / my_dpi), facecolor='white', dpi=my_dpi)

    plt.ylim(bottom=2, top=10)
    for idx, curr_method in enumerate(methods):

        color = colors[idx]
        linestyle = np.random.choice(linestyles, 1)[0]

        if all(v is None for v in results[curr_method + '_mtl']):
            continue
        mean = np.nanmean(results[curr_method + '_mtl'], axis=0)
        std = np.nanstd(results[curr_method + '_mtl'], axis=0)

        plt.plot(mean, color=color, linestyle=linestyle, linewidth=2, label=curr_method)

        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1, edgecolor=color, facecolor=color, antialiased=True)
        plt.xlabel('number of tasks', fontsize=38, fontweight="normal")
        plt.ylabel('test error', fontsize=38, fontweight="normal")
        plt.legend()

    plt.tight_layout()
    plt.savefig('temp_mtl' + '_' + str(datetime.datetime.now()).replace(':', '') + '.png', format='png')
    plt.pause(0.01)
    plt.close()
