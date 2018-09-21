import numpy as np
import matplotlib.pyplot as plt
from nudetect import GammaFlood, Source

am = Source.from_csv('Am241')

# Here, we initialize a 'GammaFlood' object. This stores information about the 
# experiment, and will be populated with analyzed data as we call its methods
# for processing the raw gamma flood data.
gamma = GammaFlood('20170315_H100_gamma_Am241_-10C_0V.fits', # raw data
    data_dir='outputs/{}/region/data', # default save path for data
    plot_dir='outputs/{}/region/plots', # default save path for plots
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

# Selecting a 6 x 6 region of the detector, in which one corner of the 
# region is at (6, 6), and the other is at (12, 12)
gamma.select_detector_region(6, 6, 13, 13)


# Loading the raw data
gamma.load_raw_data()


#
# Processing data
#

# Here we mask bursting pixels that get more than 2 sigma counts greater than
# the mean counts across the pixels. We also set the method to ignore whether
# a pulse height is positive, whereas its defualt behavior is to mask out 
# non-positive pulse heights.
gamma.gen_count_map(mask_PH=False, mask_sigma_above=2)

# Here we try to fit the 14 keV line of Am241, rather than the defualt
# 60 keV line. We also set the width of the interval in which the function
# looks for this line to be 2000 (smaller than the defualt 3000) so that
# it won't accidentally pick up the stronger 60 keV line, and set the 
# gain estimate that is used to estimate the location of the line slightly
# lower than the 0.014 kev/channel default to be extra safe.
gamma.gen_quick_gain(energy=14, search_width=2000, gain_estimate=0.013)

# Here, we reduce the energy_range from the default (0.01, 120). Maybe we want
# to focus on lower energies or something. We reduce the number of bins
# accordinly (from 10000) and save the output data file in the same place
# our plots save, cause why not, I guess.
gamma.gen_spectrum(energy_range=(0.01, 40), bins=5000, data_dir=gamma.plot_dir)


# Now, our processed data attributes have been populated with data:
#   gamma.count_map.shape: (32, 32)
#   gamma.gain.shape:      (32, 32)
#   gamma.spectrum.shape:  (2, 10000)

#
# Plotting
#


# The plotting methods called below draw data from the processed 
# data attributes populated above.

# Here, we fit the 14 keV line and display the fit on the plot, instead of 
# the default 60 keV line. Since this line has some lines close-by in the 
# high energy direction and shouldn't have as significant low-energy 
# tailing, we have modified the fitting interval appropriately with 
# 'fit_below' and 'fit_above' (which set the lower and upper bounds of
# this interval relative to the center of the peak, in channels).
gamma.plot_spectrum(energy=14, fit_below=100, fit_above=90)


# Plots a histogram that bins pixels by their event counts.
gamma.plot_pixel_hist('Count')

# Plots heatmaps of counts and gain for each pixel. Here, we specify what type
# of data we want to plot, and the method figures out the rest.
gamma.plot_pixel_map('Count')
gamma.plot_pixel_map('Gain')