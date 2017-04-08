import matplotlib.pyplot as plt
import numpy as np


def line_graph(K, values, title, filename, y, x):
    # Plot
    plt.figure(2, figsize=(7, 6))
    plt.plot(K, values, 'sb-', linewidth=3)
    plt.grid(True)
    plt.ylabel(y)
    plt.xlabel(x)
    plt.title(title)
    plt.savefig("../Figures/" + filename + ".pdf")


def bar_graph(values, title, filename):
    K = len(values)
    inds = np.arange(K)
    plt.figure(2, figsize=(6, 4))
    plt.bar(inds, values, align="center")
    plt.grid(True)
    plt.ylabel("Proportion")
    plt.xlabel("Cluster Label")
    plt.title(title)
    plt.savefig("../Figures/" + filename + ".pdf")