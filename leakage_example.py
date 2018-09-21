import numpy as np
import matplotlib.pyplot as plt
from nudetect import Leakage

leak = Leakage('/disk/lif2/spike/detectorData/H117/20180801_H117_leakage_redlen',
    plot_dir='/users/jmsander/outputs/H117/leakage/plots',
    data_dir='/users/jmsander/outputs/H117/leakage',
    detector='H117',
    temps={5, 0, -5, -10})

# Generates a map of pixel data for each combination of temperature, bias
# voltage and mode (charge-pump or normal). These maps are stored in the 
# instance attribute 'maps' as a 3D numpy.ndarray. Indexing along the 0th
# axis only will give a 2D pixel map of leakage current at a combination
# of temperature, voltage, and mode. A pandas.DataFrame object stored in 
# the instance attribute 'stats' contains mean, stddev, and outlier data
# for each of these combinations, as well as the index along the 0th axis
# of 'maps' at which this combination is represented by a pixel map. For
# more on dealing with these attributes, look at the docstrings for the
# methods 'stlice_stats' and 'slice_maps'.
leak.gen_leak_maps()

# These functions are mostly wrappers around the 'Experiment.plot_pixel_hist'
# and 'Experiment.plot_pixel_map' methods. The main difference is that they
# will plot a pixel histogram/map for all combinations of mode, temp, and
# voltage is given no arguments, or just a subset of the combinations if
# values or sets of values are supplied to the 'mode', 'temp', or 'voltage'
# arguments. See docstrings for more.
leak.plot_leak_maps(title='auto')
leak.plot_leak_hists(title='auto')

# Plots leakage current with respect to bias voltage as a line plot (with
# error bars).
leak.plot_line_current()

# Plots number of outlier pixels with respect to bias voltage as a line plot.
leak.plot_line_outliers()
