import numpy as np
import matplotlib.pyplot as plt
from nudetect import GammaFlood, Source

# Initializing a 'Source' object from a CSV file logging commonly used X-ray
# sources. In this case, we take the source that is labelled as the default
# Am241 source in this CSV. For how to modify this CSV, otherwise initialize
# 'Source' objects, and viewing emission line data for sources, call 
# 'help(Source)' for the corresponding class's documentation.
am = Source.from_csv('Am241')

# Here, we initialize a 'GammaFlood' object. This stores information about the 
# experiment, and will be populated with analyzed data as we call its methods
# for processing the raw gamma flood data.
gamma = GammaFlood('20170315_H100_gamma_Am241_-10C_0V.fits', # raw data
    data_dir='outputs/{}/data', # default save path for data
    plot_dir='outputs/{}/plots', # default save path for plots
    detector='H100', # detector ID
    source=am, # Used to fit peaks and get gain data
    voltage=0, # in volts
    temp=-10) # in degrees celsius

# Note: The '{}' in 'save_dir' is automatically formatted to the detector ID.
# In this case, data_dir == 'data/H120'

# At this point, we have some processed data attributes initialized, but not
# contianing any data:
#   gamma.count_map: None
#   gamma.gain:      None
#   gamma.spectrum:  None

# Loading the raw data
gamma.load_raw_data()


#
# Processing data
#

# Populates the attribute 'count_map', a 32 x 32 array with count data for 
# each pixel. The method will store this output in an ascii file in the 
# directory 'outputs/H100'.
gamma.gen_count_map()

# 'gain' is a 32 x 32 array with gain data for each pixel. Here, a spectrum 
# plot is saved for each individual pixel in the directory 
# 'outputs/H100/pixels', and the gain data saved to 'outputs/H100'.
gamma.gen_quick_gain(plot_subdir='pixels')

# 'spectrum' is a 2 x 10000 array representing the whole-detector spectrum.
# This output is saved to 'outputs/H100'.
gamma.gen_spectrum()


# Now, our processed data attributes have been populated with data:
#   gamma.count_map.shape: (32, 32)
#   gamma.gain.shape:      (32, 32)
#   gamma.spectrum.shape:  (2, 10000)

#
# Plotting
#


# The plotting methods called below draw data from the processed 
# data attributes populated above.

# Plots and fits 'spectrum'.
gamma.plot_spectrum()


# Plots a histogram that bins pixels by their event counts.
gamma.plot_pixel_hist('Count')

# Plots heatmaps of counts and gain for each pixel. Here, we specify what type
# of data we want to plot, and the method figures out the rest.
gamma.plot_pixel_map('Count')
gamma.plot_pixel_map('Gain')