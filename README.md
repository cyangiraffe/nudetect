# NuDetect

NuDetect uses an object-oriented framework to organize analyses of different detector experiments. For example, the class ```GammaFlood``` contains methods and takes initialization parameters specific to the analysis and plotting of gamma flood data. Other such classes include ```Noise``` (for noise data, WIP) and ```Leakage``` (for leakage current data, WIP). 

## Getting Started

### Installing

Currently, the package can be crudely installed by putting the 'nudetect.py' file into the directory in which you want to run data analysis scripts and then imported like any other module.

### Required Packages

* Numpy
* Astropy
* Scipy
* Matplotlib

**Note:** for matplotlib to work on the server lif.srl.caltech.edu (as of 8 August 2018), you must switch the matplotlib backend to 'agg'. To do this permanently in a miniconda distribution, copy the matplotlibrc sample file at miniconda3/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc to the .config/matplotlib directory, open the file (.config/matplotlib/matplotlibrc) and set the value of 'backend' to 'agg'.

### Example Script
```python
import numpy as np
import matplotlib.pyplot as plt
from nudetect import GammaFlood

# Here, we initialize a 'GammaFlood' object. This stores information about the 
# experiment, and will be populated with analyzed data as we call its methods
# for processing the raw gamma flood data.
gflood = GammaFlood('20170315_H100_gamma_Am241_-10C.0V.fits', # raw data
					detector='H100', # detector ID
					source='Am241', # Used to fit peaks and get gain data
					voltage='0V',
					temp='-10C')

#
# Processing data
#

# 'count_map' is a 32 x 32 array with count data for each pixel. As called
# here, it will store the count data in the attribute 'gflood.count_map', 
# and in an ascii file in the current directory.
gflood.count_map()

# 'gain' is a 32 x 32 array with gain data for each pixel. Here, a spectrum 
# plot is saved for each individual pixel in the directory 'pixels/H100' and 
# gain data is saved to a directory 'energy', to 'gain', and to 'gflood.gain'.
gain = gflood.quick_gain(plot_dir='pixels/H100', data_dir='energy')

# 'spectrum' is a 2 x 10000 array representing the whole-detector spectrum.
# The array is stored in 'gflood.spectrum' and a file in 'energy'.
# Since we don't supply gain data as a parameter, it takes the data from the
# attribute 'gflood.gain'.
gflood.get_spectrum(save_dir='energy')


#
# Plotting
#

# Plots and fits 'spectrum'. By default, this saves the plot to the directory
# 'energy' but doesn't plt.show() the figure. Here we fine tune how far below
# the centroid we consider for fitting with the kwarg 'fit_low'.
gflood.plot_spectrum(save_dir='energy', fit_low=77)

# Plots a histogram that bins pixels by their event counts. We've configured
# it here to not save a file, but to show the result
gflood.count_hist(save=False)
plt.show()

# Plots heatmaps of counts and gain for each pixel. The functions will save
# the figures to the current directory. The second arguments of each are used
# in labels and plot titles. Here, we _have_ to supply the array of data
# being plotted.
gflood.pixel_map(gflood.count_map, 'Counts')
gflood.pixel_map(gain, 'Gain')
```

## Developer Notes
* All classes with data analysis/plotting methods inherit from the ```Experiment``` base class, which is not useful to instantiate on its own, but contains methods to be shared with its children.
* There is one additional class called ```Line``` that can store information about a spectral line and the source that emits it.
* The ```construct_path``` method of the ```Experiment``` class is designed to throw a lot of exceptions and be strict about formatting early on to avoid complications later. Call it early in scripts to avoid losing the results of a long computation to a mistyped directory.

## Authors

* **Hiromasa Miyasaka** - *Wrote original IDL scripts*
* **Sean Pike** - *Wrote scripts in Python* - [snpike](https://github.com/snpike/)
* **Julian Sanders** - *Wrote documentation, helper functions, and classes. Edited existing code for style and efficiency.* - [cyangiraffe](https://github.com/colcaboose)
* **Andrew Sosanya** - *Documentation and logistics* - [DrewSosa](https://github.com/DrewSosa)
