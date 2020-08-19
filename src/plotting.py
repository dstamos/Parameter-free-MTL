import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib

font = {'size': 32}

matplotlib.rc('font', **font)
# matplotlib.use('Agg')


def plot_stuff(results, methods, dataset):
    linestyles = ['-', ':', '--', '-.']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black',
              'crimson', 'red', 'green', 'blue']

    my_dpi = 100
    plt.figure(figsize=(1664 / my_dpi, 936 / my_dpi), facecolor='white', dpi=my_dpi)

    # plt.ylim(bottom=8, top=12)
    if dataset == 'synthetic-regression':
        plt.ylim(bottom=3, top=5.5)
    if dataset == 'schools':
        plt.ylim(bottom=9.8, top=18)
    for idx, curr_method in enumerate(methods):
        # color = colors[idx]
        # linestyle = np.random.choice(linestyles, 1)[0]

        if curr_method == 'ITL':
            color = 'black'
            linestyle = '-'
        elif curr_method == 'Oracle':
            color = 'red'
            linestyle = '-'
        elif curr_method == 'Aggressive':
            color = 'tab:blue'
            linestyle = '-.'
        elif curr_method == 'Lazy':
            color = 'tab:green'
            linestyle = ':'
        elif curr_method == 'Aggressive_KT':
            color = 'tab:brown'
            linestyle = '-.'
        elif curr_method == 'Lazy_KT':
            color = 'tab:orange'
            linestyle = ':'
        elif curr_method == 'Aggressive_SS':
            color = 'tab:olive'
            linestyle = '-.'
        elif curr_method == 'Lazy_SS':
            color = 'tab:purple'
            linestyle = ':'

        mean = np.nanmean(results[curr_method + '_accu'], axis=0)
        std = np.nanstd(results[curr_method + '_accu'], axis=0)

        if 'Aggressive' in curr_method:
            plt.plot(mean, color=color, linestyle=linestyle, linewidth=2, label=curr_method.replace('Aggressive', 'Aggr'))
        else:
            plt.plot(mean, color=color, linestyle=linestyle, linewidth=2, label=curr_method)

        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1, edgecolor=color, facecolor=color, antialiased=True)
        plt.xlabel('iterations', fontsize=38, fontweight="normal")
        plt.ylabel('cumulative error', fontsize=38, fontweight="normal")
        plt.xticks(fontsize=36)
        # plt.legend()
        if dataset == 'synthetic-regression' and not all(['KT' not in c for c in methods]):
            plt.legend(prop={'size': 32})
        else:
            plt.legend(prop={'size': 46})


    plt.tight_layout()
    plt.savefig(dataset + '_accu' + '_' + str(datetime.datetime.now()).replace(':', '') + '.png', format='png')
    # plt.pause(0.01)
    plt.close()

    my_dpi = 100
    plt.figure(figsize=(1664 / my_dpi, 936 / my_dpi), facecolor='white', dpi=my_dpi)

    # plt.ylim(bottom=8, top=12)
    # plt.ylim(bottom=9, top=16)
    if dataset == 'schools':
        plt.ylim(bottom=9, top=15)
    for idx, curr_method in enumerate(methods):

        # color = colors[idx]
        # linestyle = np.random.choice(linestyles, 1)[0]

        if curr_method == 'ITL':
            color = 'black'
            linestyle = '-'
        elif curr_method == 'Oracle':
            color = 'red'
            linestyle = '-'
        elif curr_method == 'Aggressive':
            color = 'tab:blue'
            linestyle = '-.'
        elif curr_method == 'Lazy':
            color = 'tab:green'
            linestyle = ':'
        elif curr_method == 'Aggressive_KT':
            color = 'tab:brown'
            linestyle = '-.'
        elif curr_method == 'Lazy_KT':
            color = 'tab:orange'
            linestyle = ':'
        elif curr_method == 'Aggressive_SS':
            color = 'tab:olive'
            linestyle = '-.'
        elif curr_method == 'Lazy_SS':
            color = 'tab:purple'
            linestyle = ':'

        if all(v is None for v in results[curr_method + '_mtl']):
            continue
        mean = np.nanmean(results[curr_method + '_mtl'], axis=0)
        std = np.nanstd(results[curr_method + '_mtl'], axis=0)

        if 'Aggressive' in curr_method:
            plt.plot(mean, color=color, linestyle=linestyle, linewidth=2, label=curr_method.replace('Aggressive', 'Aggr'))
        else:
            plt.plot(mean, color=color, linestyle=linestyle, linewidth=2, label=curr_method)

        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1, edgecolor=color, facecolor=color, antialiased=True)
        plt.xlabel('number of tasks', fontsize=38, fontweight="normal")
        plt.ylabel('test error', fontsize=38, fontweight="normal")
        # plt.legend()
        if dataset == 'synthetic-regression' and not all(['KT' not in c for c in methods]):
            plt.legend(prop={'size': 32})
        else:
            plt.legend(prop={'size': 46})

    plt.tight_layout()
    plt.savefig(dataset + '_mtl' + '_' + str(datetime.datetime.now()).replace(':', '') + '.png', format='png')
    # plt.pause(0.01)
    # plt.close()


def plot_grid(grid, x_range, y_range, name, timestamp):
    max_idx = np.unravel_index(np.argmax(grid, axis=None), grid.shape)

    import matplotlib
    import warnings

    class SqueezedNorm(matplotlib.colors.Normalize):
        def __init__(self, vmin=None, vmax=None, mid=0.0, s1=2.0, s2=2.0, clip=False):
            self.vmin = vmin  # minimum value
            self.mid = mid  # middle value
            self.vmax = vmax  # maximum value
            self.s1 = s1
            self.s2 = s2
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            f = lambda x, zero, vmax, s: np.abs((x - zero) / (vmax - zero)) ** (1. / s) * 0.5
            self.g = lambda x, zero, vmin, vmax, s1, s2: f(x, zero, vmax, s1) * (x >= zero) - f(x, zero, vmin, s2) * (x < zero) + 0.5
            matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            r = self.g(value, self.mid, self.vmin, self.vmax, self.s1, self.s2)
            return np.ma.masked_array(r)

    norm = SqueezedNorm(vmin=np.nanmin(grid[:]), vmax=np.nanmax(grid[:]), mid=np.nanmedian(grid[:]), s1=0.2, s2=0.2)
    my_dpi = 100
    plt.figure(figsize=(1080 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi)
    plt.imshow(grid.T, origin='lower', extent=[np.min(x_range),
                                               np.max(x_range),
                                               np.min(y_range),
                                               np.max(y_range)], interpolation="none",
               cmap='Greens_r', aspect='auto', norm=norm)
    plt.title(name)
    plt.xlabel('inner wealth')
    plt.ylabel('meta wealth')
    plt.colorbar()
    plt.savefig('grid_' + name + '_' + str(timestamp).replace(':', '') + '.png', format='png')
    plt.close()
