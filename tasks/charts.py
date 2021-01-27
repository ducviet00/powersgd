import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use(['science','ieee','grid', 'no-latex'])
def savefig(x, y, dirname="./fig", name="test.svg"):
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(x, y)
    dirname = os.path.join(dirname, name)
    fig.savefig(dirname, facecolor='w', edgecolor='none')   # save the figure to file
    plt.close(fig)