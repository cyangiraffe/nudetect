import numpy as np
import matplotlib.pyplot as plt
from nudetect import Noise

noise = Noise('/disk/lif2/spike/detectorData/'
    'H117/20180803_H117_noise_-10C_-500V/20180803_H117_noise_-10C_-500V.fits',
    data_dir='/users/jmsander/outputs/{}/noise/data',
    plot_dir='/users/jmsander/outputs/{}/noise/plots',
    detector='H117',
    voltage=-500,
    temp=5)

noise.select_detector_region(5, 5, 10, 10)

noise.load_raw_data()

#
# Processing data
#

# Generating both count and FWHM data for each pixel. 

noise.gen_quick_noise(plot_subdir='pixels')

noise.gen_full_noise()


#
# Plotting
#

# Plotting a pixel heatmap and histogram of count data wrt pixels.
noise.plot_pixel_map('Count')
noise.plot_pixel_hist('Count')

# Plotting a pixel heatmap and histogram using the FWHM of the 
# Gaussian fitted to the noise spectrum.
noise.plot_pixel_map('FWHM')
noise.plot_pixel_hist('FWHM')


noise.plot_pixel_map('Mean')
noise.plot_pixel_hist('Mean')