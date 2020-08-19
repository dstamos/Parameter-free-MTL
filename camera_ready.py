import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib
from src.plotting import plot_stuff
import pickle

dataset = 'synthetic-regression'
# dataset = 'schools'
# dataset = 'lenk'


# methods = ['ITL', 'Oracle', 'Aggressive', 'Lazy']
# methods = ['ITL', 'Oracle', 'Aggressive', 'Lazy', 'Aggressive_KT', 'Lazy_KT']
# methods = ['ITL', 'Aggressive', 'Lazy']
methods = ['ITL', 'Aggressive', 'Lazy', 'Aggressive_KT', 'Lazy_KT']

results = pickle.load(open('results/' + str(dataset) + '.pckl', "rb"))
plot_stuff(results, methods, dataset)

k = 1