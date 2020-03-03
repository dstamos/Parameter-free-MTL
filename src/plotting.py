import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib
matplotlib.use('Agg')


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

    # norm = SqueezedNorm(vmin=np.nanmin(grid[:]), vmax=np.nanmax(grid[:]), mid=3.5, s1=1, s2=1)
    norm = SqueezedNorm(vmin=2.8, vmax=9, mid=4.5, s1=1, s2=1)
    my_dpi = 100
    plt.figure(figsize=(1080 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi)
    plt.imshow(grid.T, origin='lower', extent=[np.min(x_range),
                                               np.max(x_range),
                                               np.min(y_range),
                                               np.max(y_range)], interpolation="none",
               cmap='RdYlGn_r', aspect='auto', norm=norm)
    plt.title(name)
    plt.xlabel('inner step size')
    plt.ylabel('meta step size')
    plt.colorbar()
    plt.savefig('grid_' + name + '_' + str(timestamp).replace(':', '') + '.png', format='png')
    plt.close()
