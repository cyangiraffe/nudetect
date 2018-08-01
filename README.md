# X-ray Vision
In its current state, this module provides one working file, 'gamma.py', which can be imported and used for analysis of full-detector gamma flood data. The program is object oriented, and the user primarily interfaces with the 'GammaFlood' class, whose methods provide data analysis and plotting features.

## Installing

Currently, the package can be crudely installed by putting the 'gamma.py' file into the directory in which you want to run data analysis scripts and then imported like any other module.

## Example Script
```python
import numpy as np
import matplotlib.pyplot as plt
import gamma

# Here, we initialize a 'GammaFlood' object. This stores information about the 
# experiment, and will be populated with analyzed data as we call its methods
# for processing the raw gamma flood data.
gflood = gamma.GammaFlood('20170315_H100_gamma_Am241_-10C.0V.fits', # raw data
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
get_spectrum(save_dir='energy')


#
# Plotting
#

# Plots and fits 'spectrum'. By default, this saves the plot to the directory
# 'energy' but doesn't plt.show() the figure. Here we fine tune how far below
# the centroid we consider for fitting with the kwarg 'fit_low'.
gflood.plot_spectrum(save_dir='energy', fit_low=77)

# Plots a histogram that bins pixels by their event counts. We've configured
# it here to not save a file, but to show the result
count_hist(save=False)
plt.show()

# Plots heatmaps of counts and gain for each pixel. The functions will save
# the figures to the current directory. The second arguments of each are used
# in labels and plot titles. Here, we _have_ to supply the array of data
# being plotted.
gflood.pixel_map(gflood.count_map, 'Counts')
gflood.pixel_map(gain, 'Gain')
```

## Built With

* Astropy (v. 3.0)
* Numpy (v. 1.14)

## Authors

* **Hiromasa Miyasaka** - *Wrote original IDL scripts*
* **Sean Pike** - *Wrote scripts in Python* - [snpike](https://github.com/snpike/)
* **Julian Sanders** - *Documentation, helper functions and classes, and organization* - [colcaboose](https://github.com/colcaboose)
* **Andrew Sosanya** - *Documentation and logistics* - [DrewSosa](https://github.com/DrewSosa)

