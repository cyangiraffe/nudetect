# X-ray Vision

## Installing

Currently, the package can be crudely installed by putting the 'gamma.py' file into the directory in which you want to run data analysis scripts and then imported like any other module.

## Example Script
```python
import numpy as np
import matplotlib.pyplot as plt
import gamma

# A filepath to some raw gamma flood data
filepath = '20170315_H100_gamma_Am241_-10C.0V.fits'


### Processing data

# 'count_map' is a 32 x 32 array with count data for each pixel
count_map = count_map(filepath)

# 'gain' is a 32 x 32 array with gain data for each pixel. Here, a spectrum 
# plot is saved for each individual pixel in the directory 'pixels/H100'.
gain = quick_gain(filepath, line, save_dir='pixels/H100')

# 'spectrum' is a 2 x 10000 array representing the whole-detector spectrum
spectrum = get_spectrum(filepath, gain)

### Plotting
# Plots and fits 'spectrum'. By default, this saves the plot to our current
# directory but doesn't show the figure.
plot_spectrum(spectrum, line, filepath=filepath)

# Plots a histogram that bins pixels by their event counts. We've configured
# it here to not save a file, but to show the result
count_hist(count_map, filepath=filepath, save=False)
plt.show()

# Plots heatmaps of counts and gain for each pixel. The functions will save
# the figures to the current directory.
pixel_map(count_map, 'Counts', filepath=filepath)
pixel_map(gain, 'Gain', filepath=filepath)
```

## Built With

* Astropy (v. 3.0)
* Numpy (v. 1.14)
* Seaborn (v. 0.9)

## Authors

* **Hiromasa Miyasaka** - *Wrote original IDL scripts*
* **Sean Pike** - *Wrote scripts in Python* - [snpike](https://github.com/snpike/)
* **Julian Sanders** - *Documentation, helper functions and classes, and organization* - [colcaboose](https://github.com/colcaboose)
* **Andrew Sosanya** - *Documentation and logistics* - [DrewSosa](https://github.com/DrewSosa)

