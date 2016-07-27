import numpy as np
import os
import triangle
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.ticker import MultipleLocator

# plot posterior results

cmd = 'python posterior_plots.py -ml .405 1.864 -.333 3.439 3414 -m 0'
os.system(cmd)

cmd = 'python posterior_plots.py -ml .513 1.635 -5.997 6.70 4434 -m 1'
os.system(cmd)

cmd = 'python posterior_plots.py -ml .523 1.635 -5.942 6.694 4370 -m 2'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Onebreak_logz_" -ml .427 1.847 -.361 3.421 3447 -m 0'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Onebreak_logz_" -ml .481 1.697 -5.54 6.706 4491 -m 1'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Onebreak_logz_" -ml .509 1.614 -5.56 6.65 4297 -m 2'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Twobreak_" -ml .383 1.906 .0954 -9.697 3.246 6.675 3096 -m 0 -tb'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Twobreak_" -ml .411 1.878 .978 -8.804 3.403 6.600 3919 -m 1 -tb'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Twobreak_" -ml .420 1.859 .836 -9.167 3.553 6.804 3955 -m 2 -tb'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Twobreak_logz_" -ml .499 1.550 2.798 -.609 1.991 3.287 3476 -m 0 -tb'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Twobreak_logz_" -ml .230 3.279 1.646 -8.671 .619 6.633 4070 -m 1 -tb'
os.system(cmd)

cmd = 'python posterior_plots.py --outdir="Twobreak_logz_" -ml .244 3.241 1.536 -9.029 .685 6.577 3975 -m 2 -tb'
os.system(cmd)


    
